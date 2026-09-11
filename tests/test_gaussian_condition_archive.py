import numpy as np

from gaussian_hand.export_condition_sequences import last_frame_cache


def test_adjacent_frame_cache_copies_outputs_and_invalidates():
    calls=[]
    def reader(sequence,frame):
        calls.append((sequence,frame));return (np.ones((2,2))*frame,np.zeros((2,2)))
    read=last_frame_cache(reader)
    a=read('a',1);a[0][:]=99
    b=read('a',1)
    assert (b[0]==1).all() and len(calls)==1
    b[0][:]=88
    assert (read('a',1)[0]==1).all()
    read('a',2);read('b',2)
    assert len(calls)==3
