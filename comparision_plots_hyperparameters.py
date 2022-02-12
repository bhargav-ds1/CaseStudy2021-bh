import matplotlib.pyplot as plt
import pickle
import pandas
import numpy as np

with open('results_without_proto_hidden50_batch_2_sequences_range(2,1501,10)_same_step.pkl','rb') as f:
    d = pickle.load(f)

plt.close('all')
fig = plt.figure()
plt.plot(range(2,1501,10),d['accuracy'])
plt.xlabel("Sequence Length - actual (1000/cycle)")
plt.ylabel("accuracy")
plt.title('Sequence Length vs. accuracy - dataset(space shuttle)')
plt.savefig('Sequence-vs-accuracy-spaceshuttle-without-proto.png')
plt.close(fig)

