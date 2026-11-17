"""Recompute reported objectives from saved raw answers and endpoint usage."""
import hashlib,math
from ourwork.baseline.metrics import answer_scores
from ourwork.baseline.records import fingerprint


def audit_table(table,tokenizer):
    for cid,record in table.items():
        assert fingerprint(record['params'])==cid
        rows=record['rows'];assert len(rows)==record['qa_count'] and rows
        assert len({r['qa_id'] for r in rows})==len(rows)
        known=0
        for row in rows:
            scores=answer_scores(row['answer'],row['reference'])
            assert all(math.isclose(row[k],v,abs_tol=1e-12) for k,v in scores.items())
            context=row['context'];assert hashlib.sha256(context.encode()).hexdigest()==row['context_sha256']
            count=len(tokenizer.encode(context,add_special_tokens=False))
            assert count==row['context_tokens'] and count<=record['params']['Bc']
            if row.get('usage'):
                usage=row['usage'];assert all(type(usage[k]) is int and usage[k]>=0 for k in ['prompt_tokens','completion_tokens'])
                known+=usage['prompt_tokens']+usage['completion_tokens']
        assert known==record['known_generator_tokens']
        eligible=all(not r['error'] and r['usage'] is not None for r in rows)
        assert (record['values'] is not None)==eligible
        if eligible:
            expected=[sum(r['f1'] for r in rows)/len(rows),known/len(rows)]
            assert all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(record['values'],expected))
    return True
