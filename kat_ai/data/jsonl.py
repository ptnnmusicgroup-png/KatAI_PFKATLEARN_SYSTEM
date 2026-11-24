import json
from pathlib import Path

def read_jsonl_texts(path):
    texts=[]
    with Path(path).open("r",encoding="utf-8") as stream:
        for line_no,line in enumerate(stream,1):
            if not line.strip(): continue
            try: row=json.loads(line)
            except json.JSONDecodeError as exc: raise ValueError(f"Invalid JSON on line {line_no}") from exc
            text=row.get("text")
            if not isinstance(text,str) or not text.strip(): raise ValueError(f"Line {line_no}: expected non-empty text")
            texts.append(text.strip())
    if not texts: raise ValueError("No usable text records found")
    return texts

def train_tokenizer(input_path,output_dir,vocab_size=8000):
    import sentencepiece as spm
    if vocab_size < 100: raise ValueError("vocab_size must be at least 100")
    texts=read_jsonl_texts(input_path); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    corpus=out/"tokenizer_corpus.txt"; corpus.write_text("\n".join(texts),encoding="utf-8")
    prefix=out/"tokenizer"
    spm.SentencePieceTrainer.train(input=str(corpus),model_prefix=str(prefix),vocab_size=vocab_size,
        model_type="bpe",character_coverage=.9995,pad_id=0,unk_id=1,bos_id=2,eos_id=3,
        pad_piece="<pad>",unk_piece="<unk>",bos_piece="<bos>",eos_piece="<eos>")
    return str(prefix)+".model"
