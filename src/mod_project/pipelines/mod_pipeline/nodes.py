import os
from ultralytics import YOLO
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

TOLERANCE = 0.5


def save(G, nodes, image, class_names, denomination):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    for idx, node in enumerate(nodes):
        x, y = node['pos']
        ax.plot(x, y, 'ro', markersize=20)
        ax.text(x + 5, y, f"{class_names[node['cls_id']]}",
                color='red', fontsize=12, weight='bold')

    for i, j in G.edges():
        x1, y1 = G.nodes[i]['pos']
        x2, y2 = G.nodes[j]['pos']
        ax.plot([x1, x2], [y1, y2], 'black', linewidth=3)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"graph_{denomination}.png")
    plt.close()


def reference_load_data(reference_dataset, reference_labels):
    """
    Creates the following structure:

    references = {
        1: [
            {
                "image_filename": "1_usd_someimage1.jpg",
                "image": <image_object>,
                "labels_filename": "1_usd_someimage1.txt",
                "labels": <parsed_labels>,
            },
            ...
        ],
        2: [
            ...
        ],
        5: [
            ...
        ],
        10: [
            ...
        ],
        50: [
            ...
        ],
        100: [
            ...
        ]
    }
    """

    def read_labels(label_content):
        labels_list = []

        for line in label_content.strip().splitlines():
            cls_id, cx, cy, bw, bh = map(float, line.split())
            labels_list.append([int(cls_id), cx, cy, bw, bh])

        return labels_list

    references = {
        1: [],
        2: [],
        5: [],
        10: [],
        50: [],
        100: []
    }

    for image_filename, image in reference_dataset.items():
        labels_filename = os.path.splitext(image_filename)[0] + ".txt"
        denomination = int(labels_filename.split('_')[0])

        content = reference_labels[labels_filename]()
        references[denomination].append({
            "image_filename": image_filename,
            "image": image,
            "labels_filename": labels_filename,
            "labels": read_labels(content),
        })

    return references


def reference_to_graphs(reference_combined, distance_treshold, class_names):
    def edge_to_edge_distance(box_1, box_2):
        x1_min, y1_min, x1_max, y1_max = box_1
        x2_min, y2_min, x2_max, y2_max = box_2
        dx = max(0, max(x1_min, x2_min) - min(x1_max, x2_max))
        dy = max(0, max(y1_min, y2_min) - min(y1_max, y2_max))
        return np.sqrt(dx ** 2 + dy ** 2)

    graphs = {
        1: [],
        2: [],
        5: [],
        10: [],
        50: [],
        100: []
    }

    for denomination, data in reference_combined.items():
        for bill_variant in data:
            image = bill_variant["image"]()

            w, h = image.size
            image_diagonal = np.sqrt(w**2 + h**2)

            G = nx.Graph()
            nodes = []

            for label in bill_variant["labels"]:
                cls_id, cx, cy, bw, bh = label

                # Convert YOLO format to box
                x1 = (cx - bw / 2) * w
                x2 = (cx + bw / 2) * w
                y1 = (cy - bh / 2) * h
                y2 = (cy + bh / 2) * h

                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2

                # Normalize center
                x_norm = x_center / w
                y_norm = y_center / h

                # Pseudo-stable node ID
                pseudo_id = (int(cls_id), x_norm, y_norm)

                # Add node
                G.add_node(pseudo_id, cls_id=int(cls_id), pos=(
                    x_norm, y_norm), box=(x1, y1, x2, y2))
                nodes.append((pseudo_id, (x1, y1, x2, y2)))

            # Add edges (undirected)
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    node_i, box_i = nodes[i]
                    node_j, box_j = nodes[j]

                    dist = edge_to_edge_distance(box_i, box_j)

                    if dist < distance_treshold * image_diagonal:
                        G.add_edge(node_i, node_j, weight=dist)

            # Save graph for this denomination
            graphs[denomination].append({
                "graph": G,
                "image_filename": bill_variant["image_filename"]
            })

    return graphs


def mod_load_data(normal_dataset, normal_labels, inpainted_dataset, inpainted_labels):
    def build_pairs(images_dict, labels_dict, normal):
        data = []
        for image_filename, image in images_dict.items():
            text_filename = os.path.splitext(image_filename)[0] + ".txt"
            # if not normal:
            #     text_filename = text_filename.replace("_mask001", "")
            labels_filename, labels = text_filename, labels_dict[text_filename]

            if text_filename not in labels_dict:
                continue

            data.append({
                "image_filename": image_filename,
                "image": image,
                "labels_filename": labels_filename,
                "labels": labels,
                "normal": normal
            })

        return data

    return build_pairs(normal_dataset, normal_labels, True) + \
        build_pairs(inpainted_dataset, inpainted_labels, False)


def mod_detect_objects(mod_combined_dataset):
    model = YOLO("data/06_models/best.pt")
    labels = {}

    for element in mod_combined_dataset:
        image = element["image"]()
        results = model(image)

        bounding_boxes = []
        for box in results[0].boxes:
            cls_id = int(box.cls)
            cls_name = results[0].names[cls_id]
            confidence = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            bounding_boxes.append((
                cls_id,
                round(confidence, 6),
                round(x1, 6),
                round(y1, 6),
                round(x2, 6),
                round(y2, 6)
            ))

        labels[element["image_filename"]] = bounding_boxes

    return labels


def mod_yolo_to_graphs(mod_combined_dataset, predicted_bounding_boxes, distance_treshold, class_names):
    def edge_to_edge_distance(box_1, box_2):
        x1_min, y1_min, x1_max, y1_max = box_1
        x2_min, y2_min, x2_max, y2_max = box_2
        dx = max(0, max(x1_min, x2_min) - min(x1_max, x2_max))
        dy = max(0, max(y1_min, y2_min) - min(y1_max, y2_max))
        return np.sqrt(dx ** 2 + dy ** 2)

    graphs = {}

    for element in mod_combined_dataset:
        image = element["image"]()
        image_filename = element["image_filename"]

        w, h = image.size
        image_diagonal = np.sqrt(w**2 + h**2)

        G = nx.Graph()
        nodes = []

        for box in predicted_bounding_boxes[image_filename]:
            cls_id, confidence, x1, y1, x2, y2 = box

            # Compute center
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            # Normalize center
            x_norm = x_center / w
            y_norm = y_center / h

            # Pseudo-stable node ID
            pseudo_id = (cls_id, x_norm, y_norm)

            # Add node
            G.add_node(pseudo_id, cls_id=cls_id, confidence=confidence,
                       pos=(x_norm, y_norm), box=(x1, y1, x2, y2))
            nodes.append((pseudo_id, (x1, y1, x2, y2)))

        # Add edges (undirected)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_i, box_i = nodes[i]
                node_j, box_j = nodes[j]

                dist = edge_to_edge_distance(box_i, box_j)

                if dist < distance_treshold * image_diagonal:
                    G.add_edge(node_i, node_j, weight=dist)

        graphs[image_filename] = {
            "graph": G,
            "normal": element["normal"]
        }

    return graphs


def mod_cvat_to_graphs(mod_combined_dataset, distance_treshold, class_names):
    def edge_to_edge_distance(box_1, box_2):
        x1_min, y1_min, x1_max, y1_max = box_1
        x2_min, y2_min, x2_max, y2_max = box_2
        dx = max(0, max(x1_min, x2_min) - min(x1_max, x2_max))
        dy = max(0, max(y1_min, y2_min) - min(y1_max, y2_max))
        return np.sqrt(dx ** 2 + dy ** 2)

    def read_labels(label_content):
        labels_list = []

        for line in label_content.strip().splitlines():
            cls_id, cx, cy, bw, bh = map(float, line.split())
            labels_list.append([int(cls_id), cx, cy, bw, bh])

        return labels_list

    graphs = {}

    for element in mod_combined_dataset:
        image = element["image"]()
        image_filename = element["image_filename"]

        w, h = image.size
        image_diagonal = np.sqrt(w**2 + h**2)

        G = nx.Graph()
        nodes = []

        for label in read_labels(element["labels"]()):
            cls_id, cx, cy, bw, bh = label

            # Convert YOLO format to box
            x1 = (cx - bw / 2) * w
            x2 = (cx + bw / 2) * w
            y1 = (cy - bh / 2) * h
            y2 = (cy + bh / 2) * h

            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            # Normalize center
            x_norm = x_center / w
            y_norm = y_center / h

            # Pseudo-stable node ID
            pseudo_id = (int(cls_id), x_norm, y_norm)

            # Add node
            G.add_node(pseudo_id, cls_id=int(cls_id), pos=(
                x_norm, y_norm), box=(x1, y1, x2, y2))
            nodes.append((pseudo_id, (x1, y1, x2, y2)))

        # Add edges (undirected)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_i, box_i = nodes[i]
                node_j, box_j = nodes[j]

                dist = edge_to_edge_distance(box_i, box_j)

                if dist < distance_treshold * image_diagonal:
                    G.add_edge(node_i, node_j, weight=dist)

        graphs[image_filename] = {
            "graph": G,
            "normal": element["normal"]
        }

    return graphs


def mod_comparison(input_graphs, cvat_graphs, reference_graphs, pos_threshold, edge_threshold):
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

    def graph_to_node_set(G):
        return set(G.nodes)

    def graph_to_edge_list(G):
        edge_list = []
        for u, v in G.edges():
            edge_list.append((u, v))
        return edge_list

    def compute_edge_jaccard(input_edges, ref_edges):
        matched = 0
        ref_edges_remaining = list(ref_edges)

        for edge_in in input_edges:
            u_in, v_in = edge_in

            for i, edge_ref in enumerate(ref_edges_remaining):
                u_ref, v_ref = edge_ref

                # Undirected match, test both directions
                dist_uv = node_distance(u_in, u_ref) + \
                    node_distance(v_in, v_ref)
                dist_vu = node_distance(u_in, v_ref) + \
                    node_distance(v_in, u_ref)

                if min(dist_uv, dist_vu) < 2 * edge_threshold:
                    matched += 1
                    ref_edges_remaining.pop(i)  # Remove matched edge
                    break  # Move to next edge_in

        union = len(input_edges) + len(ref_edges) - matched
        if union == 0:
            return 1.0
        return matched / union

    def compare_graphs(G_input, G_ref):
        input_nodes = graph_to_node_set(G_input)
        ref_nodes = graph_to_node_set(G_ref)
        node_jaccard = compute_node_jaccard(input_nodes, ref_nodes)

        input_edges = graph_to_edge_list(G_input)
        ref_edges = graph_to_edge_list(G_ref)
        edge_jaccard = compute_edge_jaccard(input_edges, ref_edges)

        return node_jaccard, edge_jaccard

    results = []

    for image_filename, G_input in input_graphs.items():
        # Extract denomination from filename, e.g. "100_usd_abc.png" → 100
        denomination = int(image_filename.split('_')[0])

        if denomination not in reference_graphs:
            print(
                f"Warning: no reference graphs for denomination {denomination}")
            continue

        best_node_jaccard = -1
        best_edge_jaccard = -1
        best_ref_idx = -1

        cvat_best_node_jaccard = -1
        cvat_best_edge_jaccard = -1

        # Compare against all reference graphs of this denomination
        for ref_idx, G_ref_dict in enumerate(reference_graphs[denomination]):
            G_ref = G_ref_dict["graph"]

            # YOLO → reference
            node_jaccard, edge_jaccard = compare_graphs(
                G_input["graph"], G_ref)

            # CVAT → reference
            G_cvat = cvat_graphs[image_filename]["graph"]
            cvat_node_jaccard, cvat_edge_jaccard = compare_graphs(
                G_cvat, G_ref)

            # Update best for YOLO
            if node_jaccard > best_node_jaccard:
                best_node_jaccard = node_jaccard
                best_edge_jaccard = edge_jaccard
                best_ref_idx = ref_idx

            # Update best for CVAT
            if cvat_node_jaccard > cvat_best_node_jaccard:
                cvat_best_node_jaccard = cvat_node_jaccard
                cvat_best_edge_jaccard = cvat_edge_jaccard

        ref_images = list(reference_graphs[denomination])

        # Store result
        result = {
            "image_filename": image_filename,
            "denomination": denomination,
            # "best_ref_idx": best_ref_idx,
            "best_ref_filename": ref_images[best_ref_idx]["image_filename"],
            "node_jaccard": best_node_jaccard,
            "edge_jaccard": best_edge_jaccard,
            "cvat_node_jaccard": cvat_best_node_jaccard,
            "cvat_edge_jaccard": cvat_best_edge_jaccard,
            "normal": G_input["normal"]
        }
        results.append(result)

    # print("Comparison Report:")

    # for res in results:
    #     denomination = res["denomination"]
    #     ref_images = list(reference_graphs[denomination])
    #     ref_image = ref_images[res['best_ref_idx']]["image_filename"]

    #     print(f"{res['image_filename']} | denom {denomination} | ref {res['best_ref_idx']} | ref_image {ref_image} | "
    #           f"YOLO node Jaccard: {res['node_jaccard']:.3f} | YOLO edge Jaccard: {res['edge_jaccard']:.3f} | "
    #           f"CVAT node Jaccard: {res['cvat_node_jaccard']:.3f} | CVAT edge Jaccard: {res['cvat_edge_jaccard']:.3f} | "
    #           f"intact: {res['normal']}")

    import pandas as pd

    df = pd.DataFrame(results)
    # Optional: reorder columns for Google Sheets
    df = df[[
        "image_filename",
        "denomination",
        "best_ref_filename",
        "node_jaccard",
        "edge_jaccard",
        "cvat_node_jaccard",
        "cvat_edge_jaccard",
        "normal"
    ]]

    # Save:
    df.to_csv("mod_comparison.csv", index=False)
    print("Saved mod_comparison.csv")

    # return df
