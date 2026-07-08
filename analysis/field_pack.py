#!/usr/bin/env python3
import os, glob, json, argparse, gc
import numpy as np
from adios2 import Stream
from joblib import Parallel, delayed
from numpy.lib.format import open_memmap

p = argparse.ArgumentParser()
p.add_argument('--params', type=str, required=True)
args = p.parse_args()
params = json.loads(args.params)

folder = os.path.abspath(params['datafolder'])
njobs = params.get('njobs', 8)
xmin = params['xmin']; xlen = params['xlen']
ymin = params['ymin']; ylen = params['ylen']
tmin = params.get('tmin', 0); tlen = params.get('tlen', 1e100)
keys = params.get('keylist', ['fB1','fB2','fB3','fE1','fE2','fE3','fN_1','fN_2'])

tmax = tmin + tlen
outdir = os.path.join(folder, 'Flds/Fld_Pack')
os.makedirs(outdir, exist_ok=True)
outname = params.get('outname', f'{xmin:.1f}_ymin{ymin:.1f}_tmin{tmin:.1f}')
save_base = os.path.join(outdir, outname.replace('.npz','').replace('.npy',''))
os.makedirs(save_base, exist_ok=True)
files0 = list(np.sort(glob.glob(os.path.join(folder, 'fields/fields*'))))

def read_time(path):
    with Stream(path, 'r') as s:
        for _ in s.steps():
            return float(s.read('Time'))

def proc(path, i):
    xmax, ymax = xmin + xlen, ymin + ylen
    with Stream(path, 'r') as s:
        for _ in s.steps():
            time = float(s.read('Time'))
            if time < tmin or time > tmax:
                return None
            X1 = s.read('X1'); X2 = s.read('X2')
            ix0 = np.argmin(np.abs(X1 - xmin)); ix1 = np.argmin(np.abs(X1 - xmax))
            iy0 = np.argmin(np.abs(X2 - ymin)); iy1 = np.argmin(np.abs(X2 - ymax))
            if ix1 < ix0: ix0, ix1 = ix1, ix0
            if iy1 < iy0: iy0, iy1 = iy1, iy0
            data = {'i': i, 't': time, 'x1': X1[ix0:ix1], 'x2': X2[iy0:iy1]}
            # print(ix0,iy0,ix1,iy1)
            for k in keys:
                data[k] = s.read(k, start=[iy0, ix0], count=[iy1-iy0, ix1-ix0])
            print('fin', i, flush=True)
            return data

print(folder, flush=True)
times0 = Parallel(n_jobs=njobs, backend='loky', prefer='processes')(
    delayed(read_time)(f) for f in files0
)
times0 = np.asarray(times0)
mask = (times0 >= tmin) & (times0 <= tmax)
files = list(np.asarray(files0)[mask])
print('selected', len(files), 'of', len(files0), flush=True)
if len(files) == 0:
    raise SystemExit('no files selected')

gen = Parallel(n_jobs=njobs, backend='loky', prefer='processes', return_as='generator', pre_dispatch=njobs)(
    delayed(proc)(path, i) for i, path in enumerate(files)
)

arrs, ts, j = {}, [], 0
for s in gen:
    if s is None:
        continue
    if not arrs:
        np.save(os.path.join(save_base, 'x1.npy'), s['x1'])
        np.save(os.path.join(save_base, 'x2.npy'), s['x2'])
        for k in keys:
            a0 = np.asarray(s[k])
            arrs[k] = open_memmap(os.path.join(save_base, k + '.npy'), mode='w+', dtype=a0.dtype, shape=(len(files),) + a0.shape)
    ts.append(s['t'])
    for k in keys:
        arrs[k][j] = s[k]
    j += 1
    if j % 100 == 0:
        for a in arrs.values(): a.flush()
        print('saved', j, flush=True)

for a in arrs.values(): a.flush()
np.save(os.path.join(save_base, 't.npy'), np.asarray(ts))
with open(os.path.join(save_base, 'meta.json'), 'w') as f:
    json.dump({'keys':['t','x1','x2']+list(keys), 'time_keys':list(keys), 'n':j, 'allocated_n':len(files), 'storage':'memmap_stream'}, f)
del arrs; gc.collect()
print('fin save', save_base, flush=True)

