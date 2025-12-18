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

def get_neighborhood(G, node, distance=1):
    """Get neighborhood of a node within distance"""
    if node not in G:
        return set(), set()
    
    nodes = {node}
    edges = set()
    current_layer = {node}
    
    for _ in range(distance):
        next_layer = set()
        for n in current_layer:
            preds = set(G.predecessors(n))
            succs = set(G.successors(n))
            
            for pred in preds:
                edges.add((pred, n))
                next_layer.add(pred)
            for succ in succs:
                edges.add((n, succ))
                next_layer.add(succ)
        
        nodes.update(next_layer)
        current_layer = next_layer
    
    return nodes, edges

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
    
    # Calculate metrics once
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    total_degrees = {n: in_degrees[n] + out_degrees[n] for n in G.nodes()}
    max_degree = max(total_degrees.values()) if total_degrees else 1
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Search
        search_term = st.text_input("🔍 Search nodes", "")
        
        # Filter by degree
        min_degree = st.slider(
            "Min connections", 
            0, 
            max_degree, 
            0,
            help="Filter nodes by minimum total connections"
        )
        
        # Max nodes to show
        max_nodes = st.slider(
            "Max nodes to display",
            10,
            min(1000, G.number_of_nodes()),
            min(200, G.number_of_nodes()),
            step=10,
            help="Limit number of nodes for performance"
        )
        
        # Node selection for neighborhood
        st.divider()
        st.subheader("🎯 Explore Node")
        
        # Get filtered nodes for selection
        filtered_nodes = [
            n for n in G.nodes() 
            if total_degrees[n] >= min_degree
            and (not search_term or search_term.lower() in str(n).lower())
        ]
        
        # Sort by degree and limit
        filtered_nodes = sorted(filtered_nodes, key=lambda n: total_degrees[n], reverse=True)[:max_nodes]
        
        selected_node = st.selectbox(
            "Select node to explore",
            [""] + filtered_nodes,
            help="Choose a node to see its neighborhood"
        )
        
        if selected_node:
            neighborhood_size = st.slider(
                "Neighborhood distance",
                1, 3, 1,
                help="How many hops from selected node"
            )
            
            show_neighborhood_only = st.checkbox(
                "Show only neighborhood",
                value=True,
                help="Hide other nodes"
            )
        
        st.divider()
        st.subheader("🎨 Visual Settings")
        
        node_size_factor = st.slider("Node size", 5, 30, 15)
        show_labels = st.checkbox("Show all labels", value=False)
        physics_enabled = st.checkbox("Physics simulation", value=True)
    
    # Determine which nodes/edges to show
    if selected_node and show_neighborhood_only:
        nodes_to_show, edges_to_show = get_neighborhood(G, selected_node, neighborhood_size)
        title = f"Neighborhood of **{selected_node}** ({neighborhood_size} hop{'s' if neighborhood_size > 1 else ''})"
    else:
        nodes_to_show = set(filtered_nodes)
        edges_to_show = {
            (u, v) for u, v in G.edges() 
            if u in nodes_to_show and v in nodes_to_show
        }
        title = f"Network View"
    
    st.subheader(f"{title} — {len(nodes_to_show)} nodes, {len(edges_to_show)} edges")
    
    # Build agraph elements
    nodes = []
    edges = []
    
    # Add nodes
    for node in nodes_to_show:
        is_selected = (node == selected_node)
        is_searched = (search_term and search_term.lower() in str(node).lower())
        
        # Node size based on degree
        degree = total_degrees[node]
        size = node_size_factor + (degree / max_degree) * node_size_factor
        
        # Node color
        if is_selected:
            color = '#FF4136'  # Red
        elif is_searched:
            color = '#FF851B'  # Orange
        else:
            # Blue gradient by degree
            intensity = min(degree / max_degree, 1)
            blue_val = int(100 + 155 * intensity)
            color = f'#{50:02x}{100:02x}{blue_val:02x}'
        
        # Show label if selected, searched, or high degree
        label = str(node) if (show_labels or is_selected or is_searched or degree > max_degree * 0.3) else ""
        
        nodes.append(
            Node(
                id=str(node),
                label=label,
                size=size,
                color=color,
                title=f"{node}\nIn: {in_degrees[node]} | Out: {out_degrees[node]}",  # Tooltip
            )
        )
    
    # Add edges
    for source, target in edges_to_show:
        is_highlighted = (selected_node and (source == selected_node or target == selected_node))
        
        edges.append(
            Edge(
                source=str(source),
                target=str(target),
                color='#FF4136' if is_highlighted else '#95a5a6',
                width=3 if is_highlighted else 1,
            )
        )
    
    # Graph configuration
    config = Config(
        width="100%",
        height=700,
        directed=True,
        physics=physics_enabled,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7CA18",
        collapsible=False,
        node={
            'labelProperty': 'label',
            'renderLabel': True,
            'fontSize': 12,
            'fontColor': 'white',
        },
        link={
            'labelProperty': 'label',
            'renderLabel': False,
            'highlightColor': '#F7CA18',
        },
    )
    
    # Render graph
    return_value = agraph(nodes=nodes, edges=edges, config=config)
    
    # Show clicked node info
    if return_value:
        st.divider()
        clicked_node = return_value
        
        if clicked_node in G.nodes():
            st.subheader(f"📊 Node: {clicked_node}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("In-Degree", in_degrees.get(clicked_node, 0))
            with col2:
                st.metric("Out-Degree", out_degrees.get(clicked_node, 0))
            with col3:
                predecessors = list(G.predecessors(clicked_node))
                st.metric("Incoming", len(predecessors))
            with col4:
                successors = list(G.successors(clicked_node))
                st.metric("Outgoing", len(successors))
            
            # Show connections
            col1, col2 = st.columns(2)
            
            with col1:
                with st.expander(f"⬅️ Incoming ({len(predecessors)})", expanded=len(predecessors) <= 10):
                    if predecessors:
                        for pred in sorted(predecessors)[:50]:
                            st.text(f"• {pred}")
                        if len(predecessors) > 50:
                            st.text(f"... and {len(predecessors) - 50} more")
                    else:
                        st.text("No incoming connections")
            
            with col2:
                with st.expander(f"➡️ Outgoing ({len(successors)})", expanded=len(successors) <= 10):
                    if successors:
                        for succ in sorted(successors)[:50]:
                            st.text(f"• {succ}")
                        if len(successors) > 50:
                            st.text(f"... and {len(successors) - 50} more")
                    else:
                        st.text("No outgoing connections")
    
    # Network stats at bottom
    with st.expander("📈 Network Statistics"):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Nodes", G.number_of_nodes())
        with col2:
            st.metric("Total Edges", G.number_of_edges())
        with col3:
            st.metric("Density", f"{nx.density(G):.4f}")
        with col4:
            st.metric("Avg Degree", f"{sum(total_degrees.values())/len(total_degrees):.1f}")
        with col5:
            st.metric("Max Degree", max_degree)
        
        # Top nodes by degree
        st.subheader("🏆 Top 10 Nodes by Degree")
        top_nodes = sorted(total_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        df = pd.DataFrame(top_nodes, columns=['Node', 'Degree'])
        df['In'] = df['Node'].map(in_degrees)
        df['Out'] = df['Node'].map(out_degrees)
        st.dataframe(df, use_container_width=True)

else:
    st.info("👈 Upload a pickle file to get started")
    st.markdown("""
    ### Expected Format:
    - NetworkX Graph/DiGraph object, or
    - pandas DataFrame with edge list (source, target columns)
    
    ### Install:
    ```bash
    pip install streamlit-agraph
    ```
    """)