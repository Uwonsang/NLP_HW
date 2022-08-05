import numpy as np
import pandas as pd
d = np.load('../data/de-en/nmt_simple_len.tgt.test.npy')
print(d)
t_preds = np.load('../code/pred_ws.npy')
print(t_preds[:30])
print(len(t_preds))
tozero = len(t_preds)
df = pd.DataFrame(dict(id=['S' + (5 - len(str(i + 1))) * '0' + str(i + 1) for i in range(len(t_preds))],
                       pred=[lbl for lbl in t_preds]))
with open('pred.csv', 'w') as f:
    f.write(df.to_csv(index=False))