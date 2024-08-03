# relationship_analysis.py

import spacy
import networkx as nx
import matplotlib.pyplot as plt
import logging
import os
import pandas as pd
import argparse

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

def analyze_relationships(data, output_dir, node_size=50, node_color="skyblue", font_size=10, font_color="darkred", edge_color="gray"):
    """
    Analyzes relationships in the provided data and generates a relationship graph.

    Parameters:
    data (pd.DataFrame): DataFrame containing text data with a 'Message Content' column.
    output_dir (str): Directory to save the output graph image.
    node_size (int): Size of the nodes in the graph.
    node_color (str): Color of the nodes in the graph.
    font_size (int): Font size of the labels in the graph.
    font_color (str): Font color of the labels in the graph.
    edge_color (str): Color of the edges in the graph.

    Returns:
    None
    """
    try:
        # Initialize a new graph
        graph = nx.Graph()
        
        # Iterate over each row in the data
        for index, row in data.iterrows():
            try:
                if pd.isnull(row['Message Content']):
                    logging.warning(f"Missing 'Message Content' at row {index}")
                    continue

                # Process the text content using SpaCy
                doc = nlp(row['Message Content'])
                
                # Extract named entities
                entities = [entity.text for entity in doc.ents]
                
                # Add edges between entities in the graph using a set to avoid duplicates
                entity_pairs = set()
                for i, entity in enumerate(entities):
                    for related_entity in entities[i + 1:]:
                        pair = tuple(sorted([entity, related_entity]))
                        if pair not in entity_pairs:
                            graph.add_edge(entity, related_entity)
                            entity_pairs.add(pair)
            except Exception as row_error:
                logging.error(f"Error processing row {index}: {row_error}")
        
        # Generate layout for the graph
        pos = nx.spring_layout(graph)
        
        # Create a plot for the graph
        plt.figure(figsize=(12, 12))
        nx.draw(graph, pos, with_labels=True, node_size=node_size, node_color=node_color, font_size=font_size, font_color=font_color, edge_color=edge_color)
        
        # Save the graph as an image
        graph_path = os.path.join(output_dir, 'relationship_graph.png')
        plt.savefig(graph_path)
        
        # Log the completion message
        logging.info(f"Relationship analysis completed and graph saved to {graph_path}.")
    except Exception as e:
        logging.error(f"Error in relationship analysis: {e}")

def main():
    """
    Main function to run the relationship analysis script.
    """
    parser = argparse.ArgumentParser(description='Analyze relationships in text data and generate a relationship graph.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file containing text data.')
    parser.add_argument('output_dir', type=str, help='Directory to save the output graph image.')
    parser.add_argument('--node_size', type=int, default=50, help='Size of the nodes in the graph.')
    parser.add_argument('--node_color', type=str, default='skyblue', help='Color of the nodes in the graph.')
    parser.add_argument('--font_size', type=int, default=10, help='Font size of the labels in the graph.')
    parser.add_argument('--font_color', type=str, default='darkred', help='Font color of the labels in the graph.')
    parser.add_argument('--edge_color', type=str, default='gray', help='Color of the edges in the graph.')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the input data
    try:
        data = pd.read_csv(args.input_file)
        if data.empty:
            logging.error(f"The input file {args.input_file} is empty.")
            return
        if 'Message Content' not in data.columns:
            logging.error(f"The input file {args.input_file} does not contain the required 'Message Content' column.")
            return
    except pd.errors.EmptyDataError:
        logging.error(f"The input file {args.input_file} is empty or invalid.")
        return
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing CSV file {args.input_file}: {e}")
        return
    except Exception as e:
        logging.error(f"Error loading input file: {e}")
        return
    
    # Perform relationship analysis
    analyze_relationships(data, args.output_dir, node_size=args.node_size, node_color=args.node_color, font_size=args.font_size, font_color=args.font_color, edge_color=args.edge_color)

if __name__ == "__main__":
    main()
    logging.info("Text extraction complete.")
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
