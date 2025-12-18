import streamlit as st
import pandas as pd
import networkx as nx
import pickle
import os
from streamlit_agraph import agraph, Node, Edge, Config

# Page config
st.set_page_config(page_title="Network Explorer", layout="wide")

# Load data
@st.cache_data
def load_data(file_path):
    """Load network data from pickle file"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def extract_neighborhood(G, product, hops_in=2, hops_out=2, max_products=20):
    """
    Extract balanced neighborhood around a product
    
    Args:
        G: Full NetworkX graph
        product: Product of interest (center of neighborhood)
        hops_in: How many levels upstream (suppliers)
        hops_out: How many levels downstream (customers)
        max_products: Maximum total nodes to include
    
    Returns:
        Set of nodes, set of edges
    """
    if product not in G:
        return set(), set()
    
    visited = set([product])
    
    # Get upstream (suppliers/inputs)
    for hop in range(hops_in):
        if len(visited) >= max_products:
            break
        current_level = list(visited)
        for node in current_level:
            if len(visited) >= max_products:
                break
            predecessors = set(G.predecessors(node)) - visited
            visited.update(list(predecessors)[:max_products - len(visited)])
    
    # Get downstream (customers/outputs)
    for hop in range(hops_out):
        if len(visited) >= max_products:
            break
        current_level = list(visited)
        for node in current_level:
            if len(visited) >= max_products:
                break
            successors = set(G.successors(node)) - visited
            visited.update(list(successors)[:max_products - len(visited)])
    
    # Get all edges between visited nodes
    edges = set()
    for node in visited:
        for pred in G.predecessors(node):
            if pred in visited:
                edges.add((pred, node))
        for succ in G.successors(node):
            if succ in visited:
                edges.add((node, succ))
    
    return visited, edges

# Title
st.title('🕸️ Network Explorer')

# Load data
LOCAL_FILE = "ionetwork.pkl"
data = None

if os.path.exists(LOCAL_FILE):
    try:
        data = load_data(LOCAL_FILE)
        st.success(f"✅ Loaded: {LOCAL_FILE}")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
else:
    uploaded_file = st.file_uploader("Upload network pickle file", type=['pkl', 'pickle'])
    if uploaded_file:
        try:
            data = pickle.load(uploaded_file)
            st.success("✅ File loaded!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if data is not None:
    # Convert to NetworkX DiGraph
    if isinstance(data, (nx.Graph, nx.DiGraph)):
        G = data if isinstance(data, nx.DiGraph) else nx.DiGraph(data)
    elif isinstance(data, pd.DataFrame):
        G = nx.from_pandas_edgelist(
            data, 
            data.columns[0], 
            data.columns[1], 
            create_using=nx.DiGraph()
        )
    else:
        st.error("Unsupported format")
        st.stop()
    
    # Simple degree dict
    degrees = {n: G.in_degree(n) + G.out_degree(n) for n in G.nodes()}
    max_degree = max(degrees.values()) if degrees else 1
    
    # Get all nodes sorted by degree
    all_nodes = sorted(G.nodes(), key=lambda n: degrees[n], reverse=True)
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Node selection dropdown
        selected_node = st.selectbox(
            "Select node to explore", 
            [""] + all_nodes,
            format_func=lambda x: f"{x} ({degrees[x]} connections)" if x else "-- Choose a node --"
        )
        
        if selected_node:
            st.divider()
            st.subheader("🎯 Neighborhood")
            
            col1, col2 = st.columns(2)
            with col1:
                hops_in = st.slider("⬅️ Hops upstream", 0, 5, 2)
            with col2:
                hops_out = st.slider("➡️ Hops downstream", 0, 5, 2)
            
            max_products = st.slider("Max nodes", 10, 200, 20, step=10)
            
            st.divider()
            
            node_size = st.slider("Node size", 5, 30, 15)
            show_labels = st.checkbox("Show labels", value=False)
            physics = st.checkbox("Physics", value=True)
            
            # Show direct connection stats
            if selected_node in G:
                direct_suppliers = len(list(G.predecessors(selected_node)))
                direct_customers = len(list(G.successors(selected_node)))
                st.info(f"**Direct connections:**\n- Suppliers: {direct_suppliers}\n- Customers: {direct_customers}")
    
    # Only show graph if a node is selected
    if selected_node:
        nodes_to_show, edges_to_show = extract_neighborhood(
            G, selected_node, hops_in, hops_out, max_products
        )
        
        st.subheader(f"{len(nodes_to_show)} nodes, {len(edges_to_show)} edges")
        
        # Build graph
        nodes = []
        edges = []
        
        for node in nodes_to_show:
            is_selected = (node == selected_node)
            
            size = node_size + (degrees[node] / max_degree) * node_size
            
            if is_selected:
                color = '#FF4136'  # Red for selected
            else:
                color = '#0074D9'  # Blue for others
            
            label = str(node) if (show_labels or is_selected) else ""
            
            nodes.append(Node(
                id=str(node),
                label=label,
                size=size,
                color=color,
                title=f"{node} (connections: {degrees[node]})"
            ))
        
        for source, target in edges_to_show:
            is_highlighted = (source == selected_node or target == selected_node)
            edges.append(Edge(
                source=str(source),
                target=str(target),
                color='#FF4136' if is_highlighted else '#95a5a6',
                width=3 if is_highlighted else 1
            ))
        
        config = Config(
            width="100%",
            height=700,
            directed=True,
            physics=physics,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#F7CA18"
        )
        
        # Render
        return_value = agraph(nodes=nodes, edges=edges, config=config)
        
        # Show clicked node info
        if return_value and return_value in G.nodes():
            st.divider()
            st.subheader(f"📊 {return_value}")
            
            predecessors = list(G.predecessors(return_value))
            successors = list(G.successors(return_value))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**⬅️ Incoming ({len(predecessors)})**")
                if predecessors:
                    for p in sorted(predecessors)[:20]:
                        st.text(f"• {p}")
                    if len(predecessors) > 20:
                        st.text(f"... +{len(predecessors) - 20} more")
            
            with col2:
                st.write(f"**➡️ Outgoing ({len(successors)})**")
                if successors:
                    for s in sorted(successors)[:20]:
                        st.text(f"• {s}")
                    if len(successors) > 20:
                        st.text(f"... +{len(successors) - 20} more")
    else:
        st.info("👈 Select a node from the dropdown to explore its neighborhood")

else:
    st.info("👈 Upload a pickle file containing a NetworkX graph or edge list DataFrame")