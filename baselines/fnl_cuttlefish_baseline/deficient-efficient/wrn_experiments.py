import json

#settings = ['ACDC_%i'%n for n in [15, 48, 64]] +\
#           ['SepHashed_%.2f'%s for s in [0.05, 0.2, 0.5]] +\
settings = ['Generic_%.2f'%s for s in [0.03, 0.1, 0.24]] +\
           ['Tucker_%.2f'%s for s in [0.21, 0.41, 0.67]] +\
           ['TensorTrain_%.2f'%s for s in [0.23, 0.44, 0.7]] +\
           ['Shuffle_%i'%n for n in [1, 3, 7]]

experiments = []

import datetime
now = datetime.datetime.now()
monthday = now.strftime("%B")[:3]+"%i"%now.day

# use these settings to train WideResNets from scratch
for s in settings:
    experiment = ["python", "main.py", "cifar10", "teacher", "--conv", s,
                  "-t", "wrn_28_10.%s.%s"%(s.lower(), monthday), "--wrn_depth", "28", "--wrn_width", "10"]
    experiments.append(experiment)
# and to train WideResNets with a teacher
for s in settings:
    experiment = ["python", "main.py", "cifar10", "student", "--conv", s, "-t", "wrn_28_10.patch",
                  "-s", "wrn_28_10.%s.student.%s"%(s.lower(), monthday), "--wrn_depth", "28", "--wrn_width", "10",
                  "--alpha", "0.", "--beta", "1e3"]
    experiments.append(experiment)

with open("wrn_cifar10.json", "w") as f:
    f.write(json.dumps(experiments))
