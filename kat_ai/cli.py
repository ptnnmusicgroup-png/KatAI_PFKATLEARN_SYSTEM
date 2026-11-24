import argparse
from kat_ai.data.jsonl import train_tokenizer
from kat_ai.training.trainer import train_model
from kat_ai.inference.engine import InferenceEngine
from kat_ai.training.trainer_v1 import train_kai1
from kat_ai.inference.engine_v1 import KatAI1InferenceEngine

def main():
    parser=argparse.ArgumentParser(prog="katai")
    sub=parser.add_subparsers(dest="command",required=True)
    tok=sub.add_parser("train-tokenizer"); tok.add_argument("--input",required=True); tok.add_argument("--output",required=True); tok.add_argument("--vocab-size",type=int,default=8000)
    train=sub.add_parser("train"); train.add_argument("--data",required=True); train.add_argument("--tokenizer",required=True); train.add_argument("--output",required=True)
    train.add_argument("--steps",type=int,default=2000); train.add_argument("--batch-size",type=int,default=4); train.add_argument("--learning-rate",type=float,default=3e-4)
    train.add_argument("--d-model",type=int,default=128); train.add_argument("--n-heads",type=int,default=4); train.add_argument("--n-layers",type=int,default=4); train.add_argument("--max-seq-len",type=int,default=256)
    train.add_argument("--resume",action="store_true"); train.add_argument("--validation-interval",type=int,default=100); train.add_argument("--warmup-steps",type=int,default=100)
    train1=sub.add_parser("train-kai1"); train1.add_argument("--data",required=True); train1.add_argument("--tokenizer",required=True); train1.add_argument("--output",required=True)
    train1.add_argument("--steps",type=int,default=2000); train1.add_argument("--batch-size",type=int,default=4); train1.add_argument("--learning-rate",type=float,default=3e-4)
    train1.add_argument("--d-model",type=int,default=128); train1.add_argument("--n-heads",type=int,default=4); train1.add_argument("--n-layers",type=int,default=4); train1.add_argument("--max-seq-len",type=int,default=256)
    train1.add_argument("--resume",action="store_true"); train1.add_argument("--validation-interval",type=int,default=100); train1.add_argument("--warmup-steps",type=int,default=100)
    chat1=sub.add_parser("chat-kai1"); chat1.add_argument("--checkpoint",required=True); chat1.add_argument("--tokenizer",required=True)
    chat=sub.add_parser("chat"); chat.add_argument("--checkpoint",required=True); chat.add_argument("--tokenizer",required=True)
    args=parser.parse_args()
    if args.command=="train-tokenizer": print("Tokenizer:",train_tokenizer(args.input,args.output,args.vocab_size))
    elif args.command=="train":
        train_model(args.data,args.tokenizer,args.output,steps=args.steps,batch_size=args.batch_size,learning_rate=args.learning_rate,
            d_model=args.d_model,n_heads=args.n_heads,n_layers=args.n_layers,max_seq_len=args.max_seq_len,resume=args.resume,
            validation_interval=args.validation_interval,warmup_steps=args.warmup_steps)
    elif args.command=="train-kai1":
        result=train_kai1(args.data,args.tokenizer,args.output,steps=args.steps,batch_size=args.batch_size,learning_rate=args.learning_rate,
            d_model=args.d_model,n_heads=args.n_heads,n_layers=args.n_layers,max_seq_len=args.max_seq_len,resume=args.resume,
            validation_interval=args.validation_interval,warmup_steps=args.warmup_steps)
        print(result)
    elif args.command=="chat-kai1":
        engine=KatAI1InferenceEngine(args.checkpoint,args.tokenizer); print("KatLearn K.AI 1 — /exit to quit")
        while True:
            message=input("You> ").strip()
            if message.casefold()=="/exit": break
            print("K.AI 1>",engine.generate(message))
    else:
        engine=InferenceEngine(args.checkpoint,args.tokenizer); print("KatAI Native — /exit to quit")
        while True:
            message=input("You> ").strip()
            if message.casefold()=="/exit": break
            print("KatAI>",engine.generate(message))

if __name__=="__main__": main()
