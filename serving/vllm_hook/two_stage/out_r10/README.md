# r10 outputs

`eval_r10_<model>.json`: pre-registered (`eval_r10.py`, commit e602b03): the current recipe against the
risk and presence probes retrained with the r9 clean-depth steps as negatives, cross-fitted by user task
(split A) and by injection task (split D). CPU only, on the r4 and r9 captures (md5s in ../out_r9/README.md
and ../README.md). The pre-registered bar and its outcome per model are under "bar".
