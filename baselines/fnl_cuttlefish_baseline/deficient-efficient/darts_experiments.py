import json

#settings = ['ACDC_%i'%n for n in [6, 12]] +\
#           ['SepHashed_%.2f'%s for s in [0.09, 0.20, 0.38]] +\
settings = ['Generic_%.2f'%s for s in [0.03, 0.06, 0.12]] +\
           ['Tucker_%.2f'%s for s in [0.24, 0.37, 0.54]] +\
           ['TensorTrain_%.2f'%s for s in [0.27, 0.41, 0.59]] +\
           ['Shuffle_%i'%n for n in [1, 2, 4]]

experiments = []

import datetime
now = datetime.datetime.now()
monthday = now.strftime("%B")[:3]+"%i"%now.day

# use these settings to train WideResNets from scratch
for s in settings:
    experiment = ["python", "main.py", "cifar10", "teacher", "--conv", s,
                  "-t", "darts.%s.%s"%(s.lower(), monthday), "--network", "DARTS"]
    experiments.append(experiment)
# and to train WideResNets with a teacher
for s in settings:
    experiment = ["python", "main.py", "cifar10", "student", "--conv", s, "-t", "darts.teacher",
                  "-s", "darts.%s.student.%s"%(s.lower(), monthday), "--network", "DARTS",
                  "--alpha", "0.", "--beta", "1e3"]
    experiments.append(experiment)

with open("darts_cifar10.json", "w") as f:
    f.write(json.dumps(experiments))
