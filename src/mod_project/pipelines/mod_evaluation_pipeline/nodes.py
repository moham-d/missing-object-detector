import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def precision_recall_curve(mod_combined_dataset, reference_graphs, distance_threshold=0.06, pos_threshold=0.1):
    def read_labels(label_content):
        labels_list = []

        for line in label_content.strip().splitlines():
            cls_id, cx, cy, bw, bh = map(float, line.split())
            labels_list.append([int(cls_id), cx, cy, bw, bh])

        return labels_list

    def graph_from_labels(labels, image_size, distance_threshold=0.02):
        w, h = image_size
        image_diagonal = np.sqrt(w**2 + h**2)

        G = nx.Graph()
        nodes = []

        for label in read_labels(labels):
            cls_id, cx, cy, bw, bh = label

            x1 = (cx - bw / 2) * w
            x2 = (cx + bw / 2) * w
            y1 = (cy - bh / 2) * h
            y2 = (cy + bh / 2) * h

            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            x_norm = x_center / w
            y_norm = y_center / h

            node_id = (int(cls_id), x_norm, y_norm)

            G.add_node(node_id, cls_id=int(cls_id), pos=(
                x_norm, y_norm), box=(x1, y1, x2, y2))
            nodes.append((node_id, (x1, y1, x2, y2)))

        def edge_to_edge_distance(box1, box2):
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2
            dx = max(0, max(x1_min, x2_min) - min(x1_max, x2_max))
            dy = max(0, max(y1_min, y2_min) - min(y1_max, y2_max))
            return np.sqrt(dx**2 + dy**2)

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_i, box_i = nodes[i]
                node_j, box_j = nodes[j]

                dist = edge_to_edge_distance(box_i, box_j)

                if dist < distance_threshold * image_diagonal:
                    G.add_edge(node_i, node_j, weight=dist)

        return G

    def node_distance(nodeA, nodeB):
        clsA, xA, yA = nodeA
        clsB, xB, yB = nodeB
        if clsA != clsB:
            return np.inf
        return np.sqrt((xA - xB)**2 + (yA - yB)**2)

    def compute_node_jaccard(input_nodes, ref_nodes):
        matched = 0
        ref_nodes_remaining = set(ref_nodes)

        for node_in in input_nodes:
            best_dist = np.inf
            best_match = None
            for node_ref in ref_nodes_remaining:
                dist = node_distance(node_in, node_ref)
                if dist < pos_threshold and dist < best_dist:
                    best_dist = dist
                    best_match = node_ref
            if best_match is not None:
                matched += 1
                ref_nodes_remaining.remove(best_match)

        union = len(input_nodes) + len(ref_nodes) - matched
        if union == 0:
            return 1.0
        return matched / union

    # Now run PR curve
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    print(thresholds)
    for threshold in thresholds:
        TP = FP = FN = TN = 0
        print(threshold)
        for element in mod_combined_dataset:
            if element["normal"] == False:
                continue  # Only include normal=True
            image = element["image"]()
            image_filename = element["image_filename"]
            denomination = int(image_filename.split('_')[0])
            labels = element["labels"]()

            # Build graph from labels
            G_input = graph_from_labels(labels, image.size, distance_threshold)
            input_nodes = set(G_input.nodes)

            # Find best matching reference
            best_node_jaccard = -1

            for ref_idx, G_ref_dict in enumerate(reference_graphs[denomination]):
                G_ref = G_ref_dict["graph"]
                ref_nodes = set(G_ref.nodes)

                node_jaccard = compute_node_jaccard(input_nodes, ref_nodes)

                if node_jaccard > best_node_jaccard:
                    best_node_jaccard = node_jaccard

            # Decision
            predict = "normal" if best_node_jaccard >= threshold else "abnormal"

            # Ground truth is always "normal" here
            if predict == "normal":
                TP += 1
            else:
                FN += 1

        # Compute precision, recall
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)

        precisions.append(precision)
        recalls.append(recall)

    # Plot PR curve
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Normal=True)')
    plt.grid(True)

    # Save the figure:
    plt.savefig("pr_curve_normal_true.png", dpi=300)
    plt.close()

    print(threshold, precisions, recalls)
    # return thresholds, precisions, recalls


def pos_threshold_sweep(mod_combined_dataset, reference_graphs, distance_threshold=0.06):
    def read_labels(label_content):
        labels_list = []
        for line in label_content.strip().splitlines():
            cls_id, cx, cy, bw, bh = map(float, line.split())
            labels_list.append([int(cls_id), cx, cy, bw, bh])
        return labels_list

    def graph_from_labels(labels, image_size, distance_threshold=0.02):
        w, h = image_size
        image_diagonal = np.sqrt(w**2 + h**2)

        G = nx.Graph()
        nodes = []

        for label in read_labels(labels):
            cls_id, cx, cy, bw, bh = label

            x1 = (cx - bw / 2) * w
            x2 = (cx + bw / 2) * w
            y1 = (cy - bh / 2) * h
            y2 = (cy + bh / 2) * h

            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            x_norm = x_center / w
            y_norm = y_center / h

            node_id = (int(cls_id), x_norm, y_norm)

            G.add_node(node_id, cls_id=int(cls_id), pos=(
                x_norm, y_norm), box=(x1, y1, x2, y2))
            nodes.append((node_id, (x1, y1, x2, y2)))

        def edge_to_edge_distance(box1, box2):
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2
            dx = max(0, max(x1_min, x2_min) - min(x1_max, x2_max))
            dy = max(0, max(y1_min, y2_min) - min(y1_max, y2_max))
            return np.sqrt(dx**2 + dy**2)

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_i, box_i = nodes[i]
                node_j, box_j = nodes[j]

                dist = edge_to_edge_distance(box_i, box_j)

                if dist < distance_threshold * image_diagonal:
                    G.add_edge(node_i, node_j, weight=dist)

        return G

    def node_distance(nodeA, nodeB):
        clsA, xA, yA = nodeA
        clsB, xB, yB = nodeB
        if clsA != clsB:
            return np.inf
        return np.sqrt((xA - xB)**2 + (yA - yB)**2)

    def compute_node_jaccard(input_nodes, ref_nodes, pos_threshold):
        matched = 0
        ref_nodes_remaining = set(ref_nodes)

        for node_in in input_nodes:
            best_dist = np.inf
            best_match = None
            for node_ref in ref_nodes_remaining:
                dist = node_distance(node_in, node_ref)
                if dist < pos_threshold and dist < best_dist:
                    best_dist = dist
                    best_match = node_ref
            if best_match is not None:
                matched += 1
                ref_nodes_remaining.remove(best_match)

        union = len(input_nodes) + len(ref_nodes) - matched
        if union == 0:
            return 1.0
        return matched / union

    # Now sweep pos_threshold:
    pos_thresholds = np.linspace(0.01, 0.2, 50)  # You can adjust range!
    avg_node_jaccards = []

    for pos_threshold in pos_thresholds:
        node_jaccards = []

        for element in mod_combined_dataset:
            if element["normal"] == False:
                continue  # Only normal=True

            image = element["image"]()
            image_filename = element["image_filename"]
            denomination = int(image_filename.split('_')[0])
            labels = element["labels"]()

            # Build graph from labels
            G_input = graph_from_labels(labels, image.size, distance_threshold)
            input_nodes = set(G_input.nodes)

            # Find best matching reference
            best_node_jaccard = -1

            for ref_idx, G_ref_dict in enumerate(reference_graphs[denomination]):
                G_ref = G_ref_dict["graph"]
                ref_nodes = set(G_ref.nodes)

                node_jaccard = compute_node_jaccard(
                    input_nodes, ref_nodes, pos_threshold)

                if node_jaccard > best_node_jaccard:
                    best_node_jaccard = node_jaccard

            node_jaccards.append(best_node_jaccard)

        # Average node_jaccard for this pos_threshold:
        avg_node_jaccard = np.mean(node_jaccards)
        avg_node_jaccards.append(avg_node_jaccard)

        print(
            f"pos_threshold={pos_threshold:.4f} → avg_node_jaccard={avg_node_jaccard:.4f}")

    # Plot:
    plt.figure(figsize=(8, 6))
    plt.plot(pos_thresholds, avg_node_jaccards, marker='o')
    plt.xlabel('pos_threshold')
    plt.ylabel('Average Node Jaccard (Normal=True)')
    plt.title('Effect of pos_threshold on Node Jaccard (Normal=True)')
    plt.grid(True)

    # Save:
    plt.savefig("pos_threshold_normal_true.png", dpi=300)
    plt.close()

    print(pos_thresholds, avg_node_jaccards)
    # Return for further analysis if needed:
    # return pos_thresholds, avg_node_jaccards


def edge_threshold_sweep(mod_combined_dataset, reference_graphs, pos_threshold=0.1, distance_threshold=0.06):
    def read_labels(label_content):
        labels_list = []
        for line in label_content.strip().splitlines():
            cls_id, cx, cy, bw, bh = map(float, line.split())
            labels_list.append([int(cls_id), cx, cy, bw, bh])
        return labels_list

    def graph_from_labels(labels, image_size, distance_threshold=0.02):
        w, h = image_size
        image_diagonal = np.sqrt(w**2 + h**2)

        G = nx.Graph()
        nodes = []

        for label in read_labels(labels):
            cls_id, cx, cy, bw, bh = label

            x1 = (cx - bw / 2) * w
            x2 = (cx + bw / 2) * w
            y1 = (cy - bh / 2) * h
            y2 = (cy + bh / 2) * h

            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            x_norm = x_center / w
            y_norm = y_center / h

            node_id = (int(cls_id), x_norm, y_norm)

            G.add_node(node_id, cls_id=int(cls_id), pos=(
                x_norm, y_norm), box=(x1, y1, x2, y2))
            nodes.append((node_id, (x1, y1, x2, y2)))

        def edge_to_edge_distance(box1, box2):
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2
            dx = max(0, max(x1_min, x2_min) - min(x1_max, x2_max))
            dy = max(0, max(y1_min, y2_min) - min(y1_max, y2_max))
            return np.sqrt(dx**2 + dy**2)

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_i, box_i = nodes[i]
                node_j, box_j = nodes[j]

                dist = edge_to_edge_distance(box_i, box_j)

                if dist < distance_threshold * image_diagonal:
                    G.add_edge(node_i, node_j, weight=dist)

        return G

    def node_distance(nodeA, nodeB):
        clsA, xA, yA = nodeA
        clsB, xB, yB = nodeB
        if clsA != clsB:
            return np.inf
        return np.sqrt((xA - xB)**2 + (yA - yB)**2)

    def compute_node_jaccard(input_nodes, ref_nodes, pos_threshold):
        matched = 0
        ref_nodes_remaining = set(ref_nodes)

        for node_in in input_nodes:
            best_dist = np.inf
            best_match = None
            for node_ref in ref_nodes_remaining:
                dist = node_distance(node_in, node_ref)
                if dist < pos_threshold and dist < best_dist:
                    best_dist = dist
                    best_match = node_ref
            if best_match is not None:
                matched += 1
                ref_nodes_remaining.remove(best_match)

        union = len(input_nodes) + len(ref_nodes) - matched
        if union == 0:
            return 1.0
        return matched / union

    def compute_edge_jaccard(input_edges, ref_edges, edge_threshold):
        def edge_distance(edge_in, edge_ref):
            u_in, v_in = edge_in
            u_ref, v_ref = edge_ref

            dist1 = node_distance(u_in, u_ref) + node_distance(v_in, v_ref)
            dist2 = node_distance(u_in, v_ref) + node_distance(v_in, u_ref)

            return min(dist1, dist2)

        matched = 0
        ref_edges_remaining = list(ref_edges)

        for edge_in in input_edges:
            best_match = None
            best_score = np.inf

            for edge_ref in ref_edges_remaining:
                score = edge_distance(edge_in, edge_ref)
                if score < 2 * edge_threshold and score < best_score:
                    best_score = score
                    best_match = edge_ref

            if best_match is not None:
                matched += 1
                ref_edges_remaining.remove(best_match)

        union = len(input_edges) + len(ref_edges) - matched
        if union == 0:
            return 1.0
        return matched / union

    # Sweep edge_threshold:
    edge_thresholds = np.linspace(0.01, 0.2, 50)  # You can adjust range!
    avg_edge_jaccards = []

    for edge_threshold in edge_thresholds:
        edge_jaccards = []

        for element in mod_combined_dataset:
            if element["normal"] == False:
                continue  # Only normal=True

            image = element["image"]()
            image_filename = element["image_filename"]
            denomination = int(image_filename.split('_')[0])
            labels = element["labels"]()

            # Build graph from labels
            G_input = graph_from_labels(labels, image.size, distance_threshold)
            input_nodes = set(G_input.nodes)
            input_edges = list(G_input.edges())

            # Find best matching reference
            best_edge_jaccard = -1

            for ref_idx, G_ref_dict in enumerate(reference_graphs[denomination]):
                G_ref = G_ref_dict["graph"]
                ref_nodes = set(G_ref.nodes)
                ref_edges = list(G_ref.edges())

                # First filter by node_jaccard:
                node_jaccard = compute_node_jaccard(
                    input_nodes, ref_nodes, pos_threshold)
                if node_jaccard < 0.90:  # Optional → only consider good node matches
                    continue

                edge_jaccard = compute_edge_jaccard(
                    input_edges, ref_edges, edge_threshold)

                if edge_jaccard > best_edge_jaccard:
                    best_edge_jaccard = edge_jaccard

            if best_edge_jaccard >= 0:
                edge_jaccards.append(best_edge_jaccard)

        # Average edge_jaccard for this edge_threshold:
        avg_edge_jaccard = np.mean(edge_jaccards)
        avg_edge_jaccards.append(avg_edge_jaccard)

        print(
            f"edge_threshold={edge_threshold:.4f} → avg_edge_jaccard={avg_edge_jaccard:.4f}")

    # Plot:
    plt.figure(figsize=(8, 6))
    plt.plot(edge_thresholds, avg_edge_jaccards, marker='o')
    plt.xlabel('edge_threshold')
    plt.ylabel('Average Edge Jaccard (Normal=True)')
    plt.title(
        f'Effect of edge_threshold on Edge Jaccard (Normal=True), pos_threshold={pos_threshold}')
    plt.grid(True)

    # Save:
    plt.savefig("edge_threshold_normal_true.png", dpi=300)
    plt.close()

    print(edge_thresholds, avg_edge_jaccards)
    # return edge_thresholds, avg_edge_jaccards
