from fastnode2vec import Graph, Node2Vec
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import re
import os

class GraphConstructor:

    def __init__(self, KG_path='KG'):
        self.KG_path = KG_path
        if not os.path.exists(KG_path):
            os.makedirs(KG_path)

    def _build_graph_nodes_and_edges_for_semester(self, df, semester):
        node_dict = {}
        node_index = 0
        edges = []

        # Filter data for the semester
        semester_df = df[df['student.semester_desc'] == semester]

        print(f"Building graph for semester {semester}...")
        for _, row in tqdm(semester_df.iterrows(), total=semester_df.shape[0], desc=f"Semester {semester}"):
            # Student Node
            student_id = row['student.id']
            student_node_label = f"student_{student_id}"
            if student_node_label not in node_dict:
                node_dict[student_node_label] = node_index
                node_index += 1
            student_node_id = node_dict[student_node_label]

            # Gender Node
            gender = row['student.gender_desc']
            gender_node_label = f"gender_{gender}"
            if gender_node_label not in node_dict:
                node_dict[gender_node_label] = node_index
                node_index += 1
            gender_node_id = node_dict[gender_node_label]
            edges.append((student_node_id, gender_node_id))

            # Age Node (optional discretization)
            age = row['student.age']
            age_group = self._discretize_age(age)
            age_node_label = f"age_{age_group}"
            if age_node_label not in node_dict:
                node_dict[age_node_label] = node_index
                node_index += 1
            age_node_id = node_dict[age_node_label]
            edges.append((student_node_id, age_node_id))

            # Origin School Node
            origin = 'ITESM' if row['student_originSchool.isITESM'] == 1 else 'Non-ITESM'
            origin_node_label = f"origin_{origin}"
            if origin_node_label not in node_dict:
                node_dict[origin_node_label] = node_index
                node_index += 1
            origin_node_id = node_dict[origin_node_label]
            edges.append((student_node_id, origin_node_id))

            # Conditioned Status Node
            conditioned = 'Conditioned' if row['student.isConditioned'] == 1 else 'NotConditioned'
            conditioned_node_label = f"conditioned_{conditioned}"
            if conditioned_node_label not in node_dict:
                node_dict[conditioned_node_label] = node_index
                node_index += 1
            conditioned_node_id = node_dict[conditioned_node_label]
            edges.append((student_node_id, conditioned_node_id))

            # Subject Node
            subject = row['subject.longName']
            subject = self._sanitize_label(subject)
            subject_node_label = f"subject_{subject}"
            if subject_node_label not in node_dict:
                node_dict[subject_node_label] = node_index
                node_index += 1
            subject_node_id = node_dict[subject_node_label]

            # Grade as Edge Weight
            grade = row['student_grades.final_numeric_afterAdjustment']
            grade = grade/100.0  # Normalize to [0, 1]
            edges.append((student_node_id, subject_node_id, grade))  # Edge with weight
            

            # Competence Node
            competence = f"{row['competence.desc']}_{row['subcompetence.level_required']}_{row['subcompetence.level_assigned']}"
            competence = self._sanitize_label(competence)
            competence_node_label = f"competence_{competence}"
            if competence_node_label not in node_dict:
                node_dict[competence_node_label] = node_index
                node_index += 1
            competence_node_id = node_dict[competence_node_label]
            edges.append((student_node_id, competence_node_id))

            # Additional nodes and edges can be added similarly

        return node_dict, edges

    def _discretize_age(self, age):
        if age < 18:
            return '<18'
        elif 18 <= age <= 20:
            return '18-20'
        elif 21 <= age <= 23:
            return '21-23'
        else:
            return '>23'
        
    def _sanitize_label(self,label):
        # Replace non-alphanumeric characters with underscores
        return re.sub(r'\W+', '_', label)

    def get_embeddings_from_df(self, df, node2vec_epochs=10):
        # Prepare to collect embeddings
        student_embeddings = {}
        semesters = sorted(df['student.semester_desc'].unique())

        for semester in semesters:
            node_dict, edges = self._build_graph_nodes_and_edges_for_semester(df, semester)

            # Save node dictionary
            with open(f'{self.KG_path}/node_dict_{semester}.txt', 'w') as f:
                for node_label, node_id in node_dict.items():
                    f.write(f"{node_id}\t{node_label}\n")
            pickle.dump(node_dict, open(f'{self.KG_path}/node_dict_{semester}.pkl', 'wb'))

            # Save edges
            with open(f'{self.KG_path}/edges_{semester}.txt', 'w') as f:
                for edge in edges:
                    if len(edge) == 2:
                        f.write(f"{edge[0]}\t{edge[1]}\n")
                    elif len(edge) == 3:
                        f.write(f"{edge[0]}\t{edge[1]}\t{edge[2]}\n")
            pickle.dump(edges, open(f'{self.KG_path}/edges_{semester}.pkl', 'wb'))

            # Store embeddings
            student_embeddings[semester] = self.get_semester_embeddings(edges, node_dict, semester, node2vec_epochs)

        # Save embeddings
        with open(f'{self.KG_path}/student_embeddings.pkl', 'wb') as f:
            pickle.dump(student_embeddings, f)

        return student_embeddings
    
    def get_embeddings_from_saved_files(self, df, node_filename, edge_filename, node2vec_epochs=10):
        student_embeddings = {}
        semesters = sorted(df['student.semester_desc'].unique())
        for semester in semesters:
            # Load nodes and edges for the semester from the pickle files
            
            loaded_node_dict = pickle.load(open(f'{node_filename}_{semester}.pkl', 'rb'))
            loaded_edges = pickle.load(open(f'{edge_filename}_{semester}.pkl', 'rb'))

            # Store embeddings
            student_embeddings[semester] = self.get_semester_embeddings(loaded_edges, loaded_node_dict, semester, node2vec_epochs)

        # Save embeddings
        with open(f'{self.KG_path}/student_embeddings.pkl', 'wb') as f:
            pickle.dump(student_embeddings, f)

        return student_embeddings
    
    def construct_graph_from_files(self, node_filename, edge_filename):
        nodes = pickle.load(open(f'{node_filename}.pkl', 'rb'))
        edges = pickle.load(open(f'{edge_filename}.pkl', 'rb'))
        weighted_edges = [(edge[0], edge[1], edge[2]) for edge in edges if len(edge) == 3]
        unweighted_edges = [(edge[0], edge[1], 1.0) for edge in nodes if len(edge) == 2]
        graph = Graph(unweighted_edges + weighted_edges, directed=False, weighted=True)
        return graph
    
    def get_semester_embeddings(self, edges, nodes, semester, node2vec_epochs=10):
        # Separate weighted and unweighted edges
        weighted_edges = [(edge[0], edge[1], edge[2]) for edge in edges if len(edge) == 3]
        unweighted_edges = [(edge[0], edge[1], 1.0) for edge in nodes if len(edge) == 2]

        # Create graph
        print(f"Creating graph for semester {semester}...")
        graph = Graph(unweighted_edges + weighted_edges, directed=False, weighted=True)

        # Train Node2Vec
        print(f"Training Node2Vec for semester {semester}...")
        n2v = Node2Vec(graph, dim=128, walk_length=80, window=10, p=1, q=1, workers=4)

        n2v.train(epochs=node2vec_epochs)

        # Get embeddings for students
        embeddings = n2v.wv
        semester_student_embeddings = {}
        for node_label, node_id in nodes.items():
            if node_label.startswith('student_'):
                semester_student_embeddings[node_label] = embeddings[node_id]

        return semester_student_embeddings