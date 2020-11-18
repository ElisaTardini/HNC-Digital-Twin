from sklearn.svm import SVC
from joblib import dump, load
from histories import *
import numpy as np
from processing import *

s = './Patient twin/'
# dictionary of files containing transition models
trans_files = {
    'Prescribed Chemo (Single/doublet/triplet/quadruplet/none/NOS)': 'svm_prescribed_chemo_poly.joblib',
    'Chemo Modification (Y/N)': 'svm_chemo_modification_rbf.joblib',
    'Dose modified': 'svm_dose_modified_poly.joblib',
    'Dose delayed': 'svm_dose_delayed_rbf.joblib',
    'Dose cancelled': 'svm_dose_cancelled_rbf.joblib',
    'Regimen modification': 'svm_regimen_modification_poly.joblib',
    'DLT (Y/N)': 'svm_dlt_rbf.joblib',
    'DLT_Dermatological': 'svm_dlt_dermatological_poly.joblib',
    'DLT_Neurological': 'svm_dlt_neurological_rbf.joblib',
    'DLT_Gastrointestinal': 'svm_dlt_gastrointestinal_poly.joblib',
    'DLT_Hematological': 'svm_dlt_hematological_rbf.joblib',
    'DLT_Nephrological': 'svm_dlt_nephrological_poly.joblib',
    'DLT_Vascular': 'svm_dlt_vascular_rbf.joblib',
    'DLT_Infection (Pneumonia)': 'svm_dlt_infection_poly.joblib',
    'DLT_Other': 'svm_dlt_other_rbf.joblib',
    'DLT_Grade': 'svm_dlt_grade_poly.joblib',
    'No imaging (0=N, 1=Y)': 'svm_no_imaging_rbf.joblib',
    'CR Primary': 'svm_cr_primary_rbf.joblib',
    'CR Nodal': 'svm_cr_nodal_poly.joblib',
    'PR Primary': 'svm_pr_primary_rbf.joblib',
    'PR Nodal': 'svm_pr_nodal_rbf.joblib',
    'SD Primary': 'svm_sd_primary_rbf.joblib',
    'SD Nodal': 'svm_sd_nodal_poly.joblib',
    'CC Platinum': 'svm_cc_regimen_rbf.joblib',
    'CC Cetuximab': 'svm_cc_regimen_rbf.joblib',
    'CC Others': 'svm_cc_regimen_rbf.joblib',
    'CC modification (Y/N)': 'svm_cc_modification_rbf.joblib',
    'CR Primary 2': 'svm_cr_primary_2_poly.joblib',
    'CR Nodal 2': 'svm_cr_nodal_2_sig.joblib',
    'PR Primary 2': 'svm_pr_primary_2_poly.joblib',
    'PR Nodal 2': 'svm_pr_nodal_2_poly.joblib',
    'SD Primary 2': 'svm_sd_primary_2_rbf.joblib',
    'SD Nodal 2': 'svm_sd_nodal_2_rbf.joblib',
    'DLT_Dermatological 2': 'svm_dlt_dermatological_2_rbf.joblib',
    'DLT_Neurological 2': 'svm_dlt_neurological_2_poly.joblib',
    'DLT_Gastrointestinal 2': 'svm_dlt_gastrointestinal_2_poly.joblib',
    'DLT_Hematological 2': 'svm_dlt_hematological_2_poly.joblib',
    'DLT_Nephrological 2': 'svm_dlt_nephrological_2_poly.joblib',
    'DLT_Vascular 2': 'svm_dlt_vascular_2_rbf.joblib',
    'DLT_Other 2': 'svm_dlt_other_2_poly.joblib',
    'Overall Survival (4 Years)': 'svm_overall_survival_4_years_rbf.joblib',
    'Feeding tube 6m': 'svm_feeding_tube_rbf.joblib',
    'Aspiration rate Post-therapy': 'svm_aspiration_rate_rbf.joblib'
}

# outcome encodings in the transition models
outcome_dicts = {'Prescribed Chemo (Single/doublet/triplet/quadruplet/none/NOS)': {0: -1.0,
  1: 0.5,
  2: 0.0,
  3: 1.0},
 'Chemo Modification (Y/N)': {0: -1.0, 1: 1.0},
 'Dose modified': {0: -1.0, 1: 1.0},
 'Dose delayed': {0: -1.0, 1: 1.0},
 'Dose cancelled': {0: -1.0, 1: 1.0},
 'Regimen modification': {0: -1.0, 1: 1.0},
 'DLT (Y/N)': {0: -1.0, 1: 1.0},
 'DLT_Dermatological': {0: -1.0,
  1: -0.33333333333333337,
  2: 0.3333333333333333,
  3: 1.0},
 'DLT_Neurological': {0: -1.0,
  1: -0.33333333333333337,
  2: 0.3333333333333333,
  3: 1.0},
 'DLT_Gastrointestinal': {0: -1.0,
  1: -0.33333333333333337,
  2: 1.0,
  3: 0.3333333333333333},
 'DLT_Hematological': {0: -1.0, 1: -0.5, 2: 0.5, 3: 1.0},
 'DLT_Nephrological': {0: -1.0, 1: 1.0},
 'DLT_Vascular': {0: -1.0, 1: -0.33333333333333337, 2: 1.0},
 'DLT_Infection (Pneumonia)': {0: -1.0, 1: 1.0},
 'DLT_Other': {0: -1.0, 1: 1.0},
 'DLT_Grade': {0: -1.0, 1: 0.5, 2: -0.5, 3: 0.0, 4: 1.0},
 'No imaging (0=N, 1=Y)': {0: -1.0, 1: 1.0},
 'CR Primary': {0: -1.0, 1: 1.0},
 'CR Nodal': {0: -1.0, 1: 1.0},
 'PR Primary': {0: -1.0, 1: 1.0},
 'PR Nodal': {0: -1.0, 1: 1.0},
 'SD Primary': {0: -1.0, 1: 1.0},
 'SD Nodal': {0: -1.0, 1: 1.0},
 'CC Platinum': {0: -1.0, 1: 1.0, 2: -1.0, 3: -1.0},
 'CC Cetuximab': {0: -1.0, 1: -1.0, 2: 1.0, 3: -1.0},
 'CC Others': {0: -1.0, 1: -1.0, 2: -1.0, 3: 1.0},
 'CC modification (Y/N)': {0: -1.0, 1: 1.0},
 'CR Primary 2': {0: 1.0, 1: -1.0},
 'CR Nodal 2': {0: -1.0, 1: 1.0},
 'PR Primary 2': {0: -1.0, 1: 1.0},
 'PR Nodal 2': {0: 1.0, 1: -1.0},
 'SD Primary 2': {0: -1.0, 1: 1.0},
 'SD Nodal 2': {0: -1.0, 1: 1.0},
 'DLT_Dermatological 2': {0: -1.0, 1: 1.0},
 'DLT_Neurological 2': {0: -1.0, 1: 1.0},
 'DLT_Gastrointestinal 2': {0: -1.0, 1: 1.0},
 'DLT_Hematological 2': {0: -1.0, 1: 1.0},
 'DLT_Nephrological 2': {0: -1.0, 1: 1.0},
 'DLT_Vascular 2': {0: -1.0, 1: 1.0},
 'DLT_Other 2': {0: -1.0, 1: 1.0},
 'Overall Survival (4 Years)': {0: -1.0, 1: 1.0},
 'Feeding tube 6m': {0: -1.0, 1: 1.0},
 'Aspiration rate Post-therapy': {0: -1.0, 1: 1.0}
 }

default_outcomes = {
    A1: {
        'Prescribed Chemo (Single/doublet/triplet/quadruplet/none/NOS)': -1.0,
        'Chemo Modification (Y/N)': -1.0,
        'Dose modified': -1.0,
        'Dose delayed': -1.0,
        'Dose cancelled': -1.0,
        'Regimen modification': -1.0,
        'DLT (Y/N)': -1.0,
        'DLT_Dermatological': -1.0,
        'DLT_Neurological': -1.0,
        'DLT_Gastrointestinal': -1.0,
        'DLT_Hematological': -1.0,
        'DLT_Nephrological': -1.0,
        'DLT_Vascular': -1.0,
        'DLT_Infection (Pneumonia)': -1.0,
        'DLT_Other': -1.0,
        'DLT_Grade': -1.0
    },
    A2: {
        'CC Platinum': -1.0,
        'CC Cetuximab': -1.0,
        'CC Others': -1.0,
        'DLT_Dermatological 2': -1.0,
        'DLT_Neurological 2': -1.0,
        'DLT_Gastrointestinal 2': -1.0,
        'DLT_Hematological 2': -1.0,
        'DLT_Nephrological 2': -1.0,
        'DLT_Vascular 2': -1.0,
        'DLT_Other 2': -1.0
    },
    A3: {}
}

derived_outcomes = {
    'Dysphagia': (lambda p: max(p['Feeding tube 6m'], p['Aspiration rate Post-therapy']))
}


def load_patient_twin():
    trans_model = {}
    # importing models
    for key, value in trans_files.items():
        path = s + value
        trans_model[key] = load(path)
    return trans_model


def parse_history(history, action):
    if len(history.shape) > 1:
        sample = np.concatenate((history.to_numpy(), history.multiply(action, axis='index').to_numpy()), axis=1)

    else:
        sample = np.concatenate((history.to_numpy(), history.multiply(action, axis='index').to_numpy()),
                                axis=None).reshape(1, -1)

    return sample


def compute_outcome(history, action, model, outcome_dict):
    return outcome_dict[model.predict(parse_history(history, action))[0]]


def next_state(history, action, action_value, outcomes, trans_model):
    if np.array(outcomes).size == 1:
            outcomes = [outcomes]
    y = pd.Series(index = history.index.tolist() + [action] + outcomes, dtype='float64')
    y.loc[:] = 0
    y.loc[history.index] = history
    y.loc[action] = action_value
    for o in outcomes:
        if (o in default_outcomes[action].keys()) and (action_value == -1):
            y.loc[o] = default_outcomes[action][o]
        elif o in derived_outcomes.keys():
            y.loc[o] = derived_outcomes[o](y)
        else:
            y.loc[o] = compute_outcome(history, action_value, trans_model[o], outcome_dicts[o])
    return y


def predict(data, step):
    trans_model = load_patient_twin()
    predicted = []
    for index, row in data.iterrows():
        predicted.append(next_state(data.loc[index, histories[step - 1]], decisions[step - 1], data.loc[index, decisions[step - 1]], states[step - 1], trans_model))
    return pd.DataFrame(predicted)
