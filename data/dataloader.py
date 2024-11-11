import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

def get_dataloaders(student_subcompetence_df, embeddings, batch_size=32):
    X = []
    y = []
    
    # Map semesters to order
    semester_order = {
        'No information': 0,
        'First Semester': 1,
        'Second Semester': 2,
        'Third Semester': 3,
        'Fourth Semester': 4,
        'Fifth Semester': 5,
        'Sixth Semester': 6,
        'Seventh Semester': 7,
        'Eighth Semester': 8
    }
    ordered_semesters = sorted(embeddings.keys(), key=lambda x: semester_order.get(x, 0))
    
    for idx, semester in enumerate(ordered_semesters[:-1]):
        next_semester = ordered_semesters[idx + 1]
        current_embeddings = embeddings[semester]
        
        for student_id in student_subcompetence_df['student.id']:
            student_label = f'student_{student_id}'
            if student_label in current_embeddings:
                # Get category for the next semester
                row = student_subcompetence_df[student_subcompetence_df['student.id'] == student_id]
                if row.empty or next_semester not in row.columns:
                    continue
                category = row[next_semester].values[0]
                if pd.isna(category) or category not in ['Low (0-1)', 'Medium (1-3)', 'High (>3)']:
                    if not pd.isna(category):
                        print(f"Skipping student {student_id} because of invalid category: {category}")
                    continue
                
                # Map category to integer
                category_mapping = {'Low (0-1)': 0, 'Medium (1-3)': 1, 'High (>3)': 2}
                target_category = category_mapping[category]
                
                X.append(current_embeddings[student_label])
                y.append(target_category)
    
    X = np.array(X)
    y = np.array(y)

    print(f"Number of samples in X: {len(X)}")
    print(f"Number of samples in y: {len(y)}")

    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    full_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))

    return train_loader, test_loader, full_dataset
