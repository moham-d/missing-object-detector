import os
from ultralytics import YOLO
import networkx as nx
import numpy as np


def load_mod_data(normal_dataset, normal_labels, inpainted_dataset, inpainted_labels):
    def build_pairs(images_dict, labels_dict, normal):
        data = []
        for image_filename, image in images_dict.items():
            text_filename = os.path.splitext(image_filename)[0] + ".txt"
            if not normal:
                text_filename = text_filename.replace("_mask001", "")
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


def detect_objects(mod_combined_dataset):
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


def image_to_graph(mod_combined_dataset, predicted_bounding_boxes, distance_treshold, class_names):
    def edge_to_edge_distance(box_1, box_2):
        x1_min, y1_min, x1_max, y1_max = box_1
        x2_min, y2_min, x2_max, y2_max = box_2
        dx = max(0, max(x1_min, x2_min) - min(x1_max, x2_max))
        dy = max(0, max(y1_min, y2_min) - min(y1_max, y2_max))
        return np.sqrt(dx ** 2 + dy ** 2)

    graphs = {}

    for element in mod_combined_dataset:
        if element["normal"] == False:
            continue

        image = element["image"]()
        image_filename = element["image_filename"]

        w, h = image.size
        image_diagonal = np.sqrt(w**2 + h**2)

        G = nx.Graph()
        nodes = []
        for idx, box in enumerate(predicted_bounding_boxes[image_filename]):
            cls_id, confidence, x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            node = {
                "cls_id": cls_id,
                "confidence": confidence,
                "pos": (x_center, y_center),
                "box": (x1, y1, x2, y2)
            }

            G.add_node(idx, **node)
            nodes.append(node)

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                dist = edge_to_edge_distance(
                    nodes[i]['box'], nodes[j]['box'])
                if dist < distance_treshold * image_diagonal:
                    G.add_edge(i, j, weight=dist)

        graphs[image_filename] = G

        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # ax = plt.gca()

        # for idx, node in enumerate(nodes):
        #     x, y = node['pos']
        #     ax.plot(x, y, 'ro', markersize=20)
        #     ax.text(x + 5, y, f"{class_names[node['cls_id']]}",
        #             color='red', fontsize=12, weight='bold')

        # for i, j in G.edges():
        #     x1, y1 = G.nodes[i]['pos']
        #     x2, y2 = G.nodes[j]['pos']
        #     ax.plot([x1, x2], [y1, y2], 'black', linewidth=3)

        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()

    return graphs


def reference_patterns(predicted_bounding_boxes):
    def edge_to_edge_distance(box_1, box_2):
        x1_min, y1_min, x1_max, y1_max = box_1
        x2_min, y2_min, x2_max, y2_max = box_2
        dx = max(0, max(x1_min, x2_min) - min(x1_max, x2_max))
        dy = max(0, max(y1_min, y2_min) - min(y1_max, y2_max))
        return np.sqrt(dx ** 2 + dy ** 2)

    patterns = {}

    for image_filename, boxes in predicted_bounding_boxes.items():
        box_coords = [
            (x1, y1, x2, y2)
            for (_, _, x1, y1, x2, y2) in boxes
        ]

        distances = {}
        for i in range(len(box_coords)):
            for j in range(i + 1, len(box_coords)):
                dist = edge_to_edge_distance(box_coords[i], box_coords[j])
                distances[(i, j)] = dist

        if len(distances) > 0:
            patterns[image_filename].append(distances)

    return patterns
