"""Single-call token-chunk RAG; exact context budget and measured finite landscape."""
from pathlib import Path
import hashlib,json,time,urllib.request,uuid
from ourwork.baseline.records import atomic_json,append_jsonl,read_jsonl,fingerprint
from ourwork.baseline.metrics import answer_scores

SYSTEM='Answer using the supplied context. Return only the shortest factual answer, without explanation. If the context is insufficient, return unknown.'


def clip_context(text,budget,tokenizer):
    ids=tokenizer.encode(text,add_special_tokens=False)
    n=min(len(ids),budget)
    while n>0:
        clipped=tokenizer.decode(ids[:n],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        actual=len(tokenizer.encode(clipped,add_special_tokens=False))
        if actual<=budget:return clipped,actual
        n-=max(1,actual-budget)
    return '',0


def chunks_for(documents,c,overlap,tokenizer):
    chunks=[];stride=max(1,c-int(c*overlap))
    for doc_id,text in enumerate(documents):
        ids=tokenizer.encode(text,add_special_tokens=False)
        for start in range(0,len(ids),stride):
            part=ids[start:start+c]
            if not part:continue
            rendered=tokenizer.decode(part,skip_special_tokens=False,clean_up_tokenization_spaces=False)
            chunks.append({'doc':doc_id,'start':start,'text':rendered})
            if start+c>=len(ids):break
    if not chunks:raise ValueError('Empty grounding corpus/chunk index')
    return chunks


def dataset_data(protocol,cache):
    from datasets import load_dataset
    common={'revision':protocol['dataset_revision'],'cache_dir':str(cache)}
    qa=load_dataset(protocol['dataset_repo'],'qapairs_'+protocol['dataset_subset'],**common)
    grounding=load_dataset(protocol['dataset_repo'],'groundingdata_'+protocol['dataset_subset'],**common)
    import random
    selected={};documents={}
    for part,count in [('test',max(protocol['search_qa_stages'])),('holdout',protocol['holdout_qa_count'])]:
        available=sorted(list(qa[part]),key=lambda r:str(r['id']))
        random.Random(protocol['data_seed']).shuffle(available)
        rows=[]
        for r in available:
            if part=='holdout' and any(r['question'].strip().casefold()==x['question'].strip().casefold() for x in selected['test']):continue
            rows.append({'id':str(r['id']),'question':r['question'],'reference':str(r['answer']),
                         'qtype':r.get('type'),'difficulty':r.get('level')})
            if len(rows)==count:break
        if len(rows)!=count:raise ValueError('Not enough distinct prepared QA rows')
        selected[part]=rows;documents[part]=[r['text'] for r in grounding[part]]
    if {r['id'] for r in selected['test']} & {r['id'] for r in selected['holdout']}:
        raise ValueError('Search/holdout identifiers overlap')
    return selected,documents


class DenseRetrieval:
    def __init__(self,documents,questions,tokenizer,embedding,protocol,cache):
        self.documents=documents;self.questions=questions;self.tokenizer=tokenizer
        self.embedding=embedding;self.protocol=protocol;self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
        self.indexes={};self.queries={}
    def retrieve(self,part,question,params):
        import numpy as np
        c=params['c'];key=(part,c)
        if key not in self.indexes:
            chunks=chunks_for(self.documents[part],c,self.protocol['chunk_overlap_fraction'],self.tokenizer)
            identity=fingerprint({'texts':[r['text'] for r in chunks],'embedding':self.protocol['embedding_revision'],'c':c})
            path=self.cache/(identity+'.npy')
            if path.exists():vectors=np.load(path,allow_pickle=False)
            else:
                vectors=self.embedding.encode([r['text'] for r in chunks],batch_size=32,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True)
                np.save(path,np.asarray(vectors,dtype=np.float32),allow_pickle=False)
            if vectors.shape!=(len(chunks),384) or not np.isfinite(vectors).all():raise ValueError('Invalid embedding matrix')
            self.indexes[key]=(chunks,vectors)
        chunks,vectors=self.indexes[key]
        qkey=(part,question)
        if qkey not in self.queries:
            self.queries[qkey]=self.embedding.encode(['Represent this sentence for searching relevant passages: '+question],
                normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)[0]
        scores=vectors@self.queries[qkey]
        positions=np.argsort(-scores,kind='stable')[:params['k']]
        text='\n\n'.join(f'[chunk {int(i)} | doc {chunks[int(i)]["doc"]}]\n{chunks[int(i)]["text"]}' for i in positions)
        context,tokens=clip_context(text,params['Bc'],self.tokenizer)
        return context,{'retrieved_chunks':[int(i) for i in positions],'context_tokens':tokens,
                        'requested_context_budget':params['Bc'],'available_chunks':len(chunks)}


class Generator:
    def __init__(self,protocol,out):
        self.protocol=protocol;self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.cache={r['key']:r for r in read_jsonl(self.out/'response_cache.jsonl')}
    def answer(self,part,qa,context):
        body={'model':self.protocol['model_alias'],'temperature':0,'top_p':1,
              'max_tokens':self.protocol['max_output_tokens'],'stream':False,
              'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':f'Context:\n{context}\n\nQuestion: {qa["question"]}\nAnswer:'}]}
        key=fingerprint({'partition':part,'qa_id':qa['id'],'model_revision':self.protocol['model_revision'],'request':body})
        if key in self.cache:return dict(self.cache[key],cache_hit=True)
        call_id=uuid.uuid4().hex
        append_jsonl(self.out/'requests.jsonl',{'call_id':call_id,'key':key,'partition':part,'qa_id':qa['id'],'request':body,'time':time.time()})
        start=time.monotonic();answer='';usage=None;error=None;payload=None
        try:
            request=urllib.request.Request('http://127.0.0.1:8765/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=self.protocol['request_timeout']) as response:payload=json.load(response)
            answer=payload['choices'][0]['message'].get('content') or ''
            usage=payload.get('usage')
            if not answer.strip() or not isinstance(usage,dict) or not all(type(usage.get(k)) is int and usage[k]>=0 for k in ['prompt_tokens','completion_tokens']):
                raise ValueError('Empty answer or unknown endpoint token usage')
        except Exception as e:error=f'{type(e).__name__}: {e}'
        result={'key':key,'call_id':call_id,'answer':answer,'usage':usage,'error':error,'response':payload,
                'elapsed_seconds':time.monotonic()-start,'cache_hit':False}
        append_jsonl(self.out/'calls.jsonl',result);append_jsonl(self.out/'response_cache.jsonl',result);self.cache[key]=result
        return result


def measure_config(params,part,qa_rows,retrieval,generator,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    stored={r['qa_id']:r for r in read_jsonl(out/'questions.jsonl')}
    for qa in qa_rows:
        if qa['id'] in stored:continue
        start=time.monotonic();context,info=retrieval.retrieve(part,qa['question'],params)
        response=generator.answer(part,qa,context)
        scores=answer_scores(response['answer'],qa['reference'])
        row={'qa_id':qa['id'],'question':qa['question'],'reference':qa['reference'],'qtype':qa.get('qtype'),
             'difficulty':qa.get('difficulty'),'answer':response['answer'],'error':response['error'],
             'usage':response['usage'],'call_id':response['call_id'],'response_cache_key':response['key'],
             'response_cache_hit':response['cache_hit'],'logical_generation_seconds':response['elapsed_seconds'],
             'measurement_wall_seconds':time.monotonic()-start,'context':context,'context_sha256':hashlib.sha256(context.encode()).hexdigest(),
             **info,**scores}
        append_jsonl(out/'questions.jsonl',row);stored[qa['id']]=row
    rows=[stored[q['id']] for q in qa_rows]
    valid=all(not r['error'] and r['usage'] is not None for r in rows)
    known=sum(r['usage']['prompt_tokens']+r['usage']['completion_tokens'] for r in rows if r['usage'])
    result={'params':params,'values':[sum(r['f1'] for r in rows)/len(rows),known/len(rows)] if valid else None,
            'exact_match':sum(r['exact_match'] for r in rows)/len(rows) if valid else None,
            'known_generator_tokens':known,'qa_count':len(rows),'rows':rows}
    atomic_json(out/'result.json',result)
    return result
