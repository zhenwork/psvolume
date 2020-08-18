import os,sys


def argument_decode(args):
    # --x y=10 z=20 --a b=10 c=20
    for key in args:
        if getattr(args,key) is None:
            continue
        elif len(getattr(args,key))==0:
            setattr(args,key,{})
        elif len(getattr(args,key))>=1:
            for key_inner in getattr(args,key):
                setattr()
    return args