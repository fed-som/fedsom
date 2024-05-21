import matplotlib.pyplot as plt
import umap    
import numpy as np    
import torch
import plotly.graph_objects as go
import matplotlib.style as style
import torch    
style.use("ggplot") 



def plot_graph(G,node_labels=None):

    pos = {(x, y, z): (x, y, z) for x, y, z in G.nodes()}
    if not node_labels:
        random_letters = ['ransomware','worm','trojan']
        node_labels = {node: random.choice(random_letters) for node in G.nodes()}

    node_x = []
    node_y = []
    node_z = []
    for node, label in node_labels.items():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        G.nodes[node]['label'] = label

    edge_x = []
    edge_y = []
    edge_z = []
    for edge in G.edges():
        start, end = edge
        start_x, start_y, start_z = pos[start]
        end_x, end_y, end_z = pos[end]
        edge_x.extend([start_x, end_x, None])
        edge_y.extend([start_y, end_y, None])
        edge_z.extend([start_z, end_z, None])

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode="markers+text",
        marker=dict(size=7, color="royalblue"), #skyblue royalblue #00e6b8  10
        text=[f"{node_labels[node]}" for node in G.nodes()],
        textposition="bottom center",
        hoverinfo="none",
        textfont_size=18 # 22
    ))

    weights = []
    for _,_,data in G.edges(data=True):
        weights.append(data['weight'])
    weights = np.array(weights)
    max_weight = weights.max()
    min_weight = weights.min()

    for i,(start,end,data) in enumerate(G.edges(data=True)):
        start_x, start_y, start_z = pos[start]
        end_x, end_y, end_z = pos[end]
       
        weight = data['weight']
        width = 20 * weight  # Adjust this factor as needed
        width = np.exp(2*(1-(weight-min_weight)/max_weight))

        fig.add_trace(go.Scatter3d(
            x=[start_x, end_x],
            y=[start_y, end_y],
            z=[start_z, end_z],
            mode="lines",
            line=dict(width=width, color="gray"),  # Set thickness based on edge weight
            hoverinfo="none"
        ))

    fig.update_layout(
        title="4x4x4 Grid Graph",
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z")
        ),
        showlegend=False
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False,title=''),
            yaxis=dict(showticklabels=False,title=''),
            zaxis=dict(showticklabels=False,title=''),
        )
    )

    fig.show()








def scatter_plot(X, labels, best_params_string, filepath):

    if X.shape[1]>2:
        umap_model = umap.UMAP(n_components=2,random_state=42)
        if isinstance(X,torch.Tensor):
            X = umap_model.fit_transform(X.cpu().detach().numpy())
        else:
            X = umap_model.fit_transform(X)

    if isinstance(labels[0],str):
        labels_dict = dict(zip(sorted(list(set(labels))),range(len(set(labels)))))
        labels_numerical = np.array([labels_dict[key] for key in labels])
    else:
        if isinstance(labels[0],torch.Tensor):
            labels = [label.item() for label in labels]
        labels_numerical = labels

    fig = plt.figure(figsize=(14, 8))
    plt.scatter(X[:, 0], X[:, 1], c=labels_numerical, cmap="tab20")  #'viridis')

    for i, label in enumerate(labels):
        if np.random.rand()<0.1:
            plt.annotate(label, (X[i,0], X[i,1]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title(best_params_string, fontsize=20, fontweight="bold")
    plt.colorbar(label="Cluster")
    plt.savefig(filepath)
    plt.close()
    del fig

def tensor_to_image_array(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    elif not isinstance(tensor, np.ndarray):
        raise ValueError("Input must be a PyTorch tensor or NumPy array")

    if tensor.ndim != 2:
        raise ValueError("Input tensor must be 2-dimensional")

    tensor = (tensor * 255).astype(np.uint8)
    img = np.stack((tensor,) * 3, axis=-1)  # Convert to 3-channel grayscale image

    return img

def vector_to_square_tensor(vector):

    vector = torch.tensor(vector) if isinstance(vector,np.ndarray) else vector
    size = int(vector.numel() ** 0.5)
    return torch.reshape(vector,(size,size))

def plot_images(vectors, rows, cols, node_coords, labels, dataset=None, savepath=None):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    for i, vector in enumerate(vectors):
        if dataset=='cifar10':
            image = vector.reshape(32,32,3)
            image = np.dot(image[...,:3],[0.2989,0.5870,0.1140])
        else:
            tensor = vector_to_square_tensor(vector)
            image = tensor_to_image_array(tensor)
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        # ax.set_title(labels[i],pad=10)
        ax.text(0.5,-0.1,f'{labels[i]}: {node_coords[i]}',horizontalalignment='center',verticalalignment='center',transform=ax.transAxes,fontsize=12)
    if savepath:
        plt.savefig(savepath)
    plt.close()






if __name__=='__main__':

    # Create a sample PyTorch tensor with random values between 0 and 1
    sample_tensor = torch.rand((256, 256))

    # Convert tensor to grayscale image
    img = tensor_to_image_array(sample_tensor)

    results_dir = Path('../../../sandbox/results/images/testing/')
    if not os.path.exists(results_dir):
        results_dir.mkdir(parents=True)  

    savepath = results_dir / 'test_image.png'
    image_array_to_image(img,savepath,show=False)

    # plt.figure(figsize=(14,8))
    # # Display the image
    # plt.imshow(img, cmap='gray')
    # plt.axis('off')
    # plt.show(block=False)

    # input('DONE')
    # plt.close()





























