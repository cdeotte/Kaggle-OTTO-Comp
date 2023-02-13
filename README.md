# Kaggle-OTTO-Comp

```
├── train
│   ├── covisit_matrices         # Compute matrices with RAPIDS cuDF
│   ├── candidates               # Generate candidates from matrices
│   ├── item_user_features       # Feature engineering with RAPIDS cuDF
│   ├── make_parquets            # Combine candidates, features, targets
│   └── ranker_models            # Train XGB model
├── infer        
│   ├── covisit_matrices_LB      # Compute matrices with RAPIDS cuDF
│   ├── candidates_LB            # Generate candidates from matrices
│   ├── item_user_features_LB    # Feature engineering with RAPIDS cuDF
│   ├── make_parquets_LB         # Combine candidates, features, targets
│   └── inference_LB             # Infer XGB model with RAPIDS FIL
├── data    
│   ├── make_train_valid.ipynb   # Run to download data
│   ├── train_data               # Train data downloaded to here
│   ├── infer_data               # Infer data downloaded to here
│   ├── covisit_matrices         # Matrices stored here
│   ├── candidate_scores         # Candidate lists and scores here
│   ├── item_user_features       # Item and user features here
│   ├── train_with_features      # Train data with features merged
│   ├── infer_with_features      # Infer data with features merged
│   ├── models                   # Trained models here
│   ├── submission_parts         # Partial submission.csv here
│   └── submission_final         # Final submission.csv here
└── README.md
```
