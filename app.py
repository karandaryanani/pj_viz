import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from st_cytoscape import cytoscape
import io
import re
import pickle

# Load data
@st.cache_data
def load_data(file):
    """Load network data from pickle file"""
    return pickle.load(file)

# Streamlit app
st.title('Network Visualization')

# File uploader
uploaded_file = st.file_uploader("Upload your network pickle file", type=['pkl', 'pickle'])

if uploaded_file is not None:
    # Load the network
    try:
        data = load_data(uploaded_file)
        
        # Check if data is a NetworkX graph or edge list
        if isinstance(data, nx.Graph) or isinstance(data, nx.DiGraph):
            DG_full = data
            # Extract edge list for other operations
            edge_df = pd.DataFrame(DG_full.edges(), columns=['source', 'target'])
        elif isinstance(data, pd.DataFrame):
            edge_df = data
            # Ensure column names are 'source' and 'target'
            if len(edge_df.columns) >= 2:
                edge_df.columns = ['source', 'target'] + list(edge_df.columns[2:])
            DG_full = nx.from_pandas_edgelist(edge_df, 'source', 'target', create_using=nx.DiGraph())
        else:
            st.error("Unsupported data format. Please upload a NetworkX graph or DataFrame with edge list.")
            st.stop()
        
        st.success(f"Network loaded successfully! Nodes: {DG_full.number_of_nodes()}, Edges: {DG_full.number_of_edges()}")
        
        # View selection
        view = st.radio('Select view', ['Whole Network', 'Sankey Diagram'])
        
        if view == 'Whole Network':
            st.header("Network Visualization")
            
            # Add a search bar and search type selection
            col1, col2 = st.columns([3, 1])
            with col1:
                search_query = st.text_input("Search for a node", "")
            with col2:
                search_type = st.radio("Search type", ["Filter", "Highlight"])
            
            # Filter or highlight based on search
            if search_query:
                escaped_query = re.escape(search_query)
                case_insensitive_pattern = f"(?i){escaped_query}"
                if search_type == "Filter":
                    # Filter the graph based on the search query
                    matching_nodes = [node for node in DG_full.nodes() if re.search(case_insensitive_pattern, str(node))]
                    DG = nx.subgraph(DG_full, matching_nodes).copy()
                    # Include neighboring nodes to show connections
                    neighbors = set()
                    for node in list(DG.nodes()):
                        neighbors.update(DG_full.predecessors(node))
                        neighbors.update(DG_full.successors(node))
                    DG = DG_full.subgraph(set(DG.nodes()).union(neighbors)).copy()
                else:
                    DG = DG_full
            else:
                DG = DG_full
            
            # Calculate in-degrees for the current graph
            in_degrees = dict(DG.in_degree())
            max_in_degree = max(in_degrees.values()) if in_degrees else 1
            
            # Calculate degree threshold for top 5%
            if len(in_degrees) > 0:
                degree_threshold = sorted(in_degrees.values(), reverse=True)[max(0, int(len(in_degrees) * 0.05))]
            else:
                degree_threshold = 0
            
            # Prepare data for Cytoscape
            nodes = []
            for node in DG.nodes():
                highlight = False
                if search_query and search_type == "Highlight":
                    if re.search(case_insensitive_pattern, str(node)):
                        highlight = True
                nodes.append({
                    'data': {
                        'id': str(node),
                        'label': str(node),
                        'size': 20 + (in_degrees[node] / max_in_degree) * 50 if max_in_degree > 0 else 20,
                        'degree': in_degrees[node],
                        'show_label': in_degrees[node] >= degree_threshold,
                        'highlight': highlight
                    }
                })
            
            edges = []
            for source, target in DG.edges():
                highlight = False
                if search_query and search_type == "Highlight":
                    if any(re.search(case_insensitive_pattern, str(n)) for n in [source, target]):
                        highlight = True
                edges.append({
                    'data': {
                        'source': str(source),
                        'target': str(target),
                        'id': f"{source}->{target}",
                        'highlight': highlight
                    }
                })
            
            elements = nodes + edges
            
            # Create Cytoscape stylesheet
            stylesheet = [
                {
                    'selector': 'node',
                    'style': {
                        'background-color': '#11479e',
                        'width': 'data(size)',
                        'height': 'data(size)',
                    }
                },
                {
                    'selector': 'node[?highlight]',
                    'style': {
                        'background-color': '#ff0000',
                    }
                },
                {
                    'selector': 'node[?show_label]',
                    'style': {
                        'content': 'data(label)',
                        'font-size': '12px',
                        'text-valign': 'bottom',
                        'text-halign': 'center',
                        'color': 'white',
                        'text-background-color': 'white',
                        'text-background-opacity': 0.7,
                        'text-background-padding': '3px',
                        'text-border-opacity': 0,
                    }
                },
                {
                    'selector': 'node:selected',
                    'style': {
                        'content': 'data(label)',
                        'font-size': '14px',
                        'text-valign': 'top',
                        'text-halign': 'center',
                        'color': 'black',
                        'text-background-color': 'white',
                        'text-background-opacity': 1,
                        'text-background-padding': '3px',
                        'text-border-opacity': 1,
                        'text-border-width': 1,
                        'text-border-color': 'black',
                    }
                },
                {
                    'selector': 'edge',
                    'style': {
                        'width': 1,
                        'line-color': '#9dbaea',
                        'target-arrow-color': '#9dbaea',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                    }
                },
                {
                    'selector': 'edge[?highlight]',
                    'style': {
                        'line-color': '#ff0000',
                        'target-arrow-color': '#ff0000',
                    }
                }
            ]
            
            # Layout configuration
            layout = {
                "name": "cose",
                "idealEdgeLength": 100,
                "nodeOverlap": 20,
                "refresh": 20,
                "fit": True,
                "padding": 30,
                "randomize": True,
                "componentSpacing": 100,
                "nodeRepulsion": 400000,
                "edgeElasticity": 100,
                "nestingFactor": 5,
                "gravity": 80,
                "numIter": 1000,
                "initialTemp": 200,
                "coolingFactor": 0.95,
                "minTemp": 1.0
            }
            
            # Render the Cytoscape component
            selected = cytoscape(
                elements,
                stylesheet,
                layout=layout,
                key="cytoscape",
                height="600px",
                width="100%",
                selection_type="single"
            )
            
            # Display selected nodes and edges
            if selected:
                st.write("**Selected nodes:**", ", ".join(selected.get('nodes', [])))
                st.write("**Selected edges:**", ", ".join(selected.get('edges', [])))
            
            # Network statistics
            st.subheader("Network Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nodes", DG.number_of_nodes())
            with col2:
                st.metric("Edges", DG.number_of_edges())
            with col3:
                st.metric("Density", f"{nx.density(DG):.4f}")
                
        elif view == 'Sankey Diagram':
            st.header("Sankey Diagram")
            
            # Prepare data for Sankey diagram
            sankey_data = edge_df.copy()
            sankey_data.columns = ['source', 'target'] if len(sankey_data.columns) == 2 else list(sankey_data.columns)
            sankey_data['weight'] = 1
            
            # Optional: filter by node if needed
            filter_option = st.checkbox("Filter by specific node")
            if filter_option:
                all_nodes = list(set(sankey_data['source'].tolist() + sankey_data['target'].tolist()))
                selected_node = st.selectbox("Select a node to focus on", sorted(all_nodes))
                
                # Filter to show only edges connected to selected node
                sankey_data = sankey_data[
                    (sankey_data['source'] == selected_node) | 
                    (sankey_data['target'] == selected_node)
                ]
            
            # Create Sankey diagram
            all_nodes = list(set(sankey_data['source'].tolist() + sankey_data['target'].tolist()))
            node_indices = {node: index for index, node in enumerate(all_nodes)}
            
            fig = go.Figure(data=[go.Sankey(
                node = dict(
                    pad = 15,
                    thickness = 20,
                    line = dict(color = "black", width = 0.5),
                    label = all_nodes,
                    color = "blue"
                ),
                link = dict(
                    source = [node_indices[source] for source in sankey_data['source']],
                    target = [node_indices[target] for target in sankey_data['target']],
                    value = sankey_data['weight']
                )
            )])
            
            fig.update_layout(title_text="Network Sankey Diagram", font_size=10, height=600)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Save Sankey as HTML
            buffer = io.StringIO()
            fig.write_html(buffer)
            html_bytes = buffer.getvalue().encode()
            
            # Download button for Sankey HTML
            st.download_button(
                label="Download Sankey HTML",
                data=html_bytes,
                file_name="sankey_diagram.html",
                mime="text/html"
            )
            
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.info("Please ensure your pickle file contains either a NetworkX graph or a pandas DataFrame with edge list.")
else:
    st.info("Please upload a pickle file to visualize the network.")
    st.markdown("""
    ### Expected Format:
    Your pickle file should contain one of the following:
    - A NetworkX Graph or DiGraph object
    - A pandas DataFrame with at least two columns (source, target) representing edges
    """)
