"""
MASKS RPG RAG System with Rich Metadata and Graph-Ready Structure
Designed for easy migration to GraphRAG
"""
import json
import logging
import os.path
import re
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import PyPDF2
import chromadb
import pandas as pd
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of content we expect to find in MASKS"""
    CORE_PHILOSOPHY = "core_philosophy"
    TECHNIQUE = "technique"
    MECHANIC = "mechanic"
    PLAYBOOK = "playbook"
    PLAYBOOK_MOVE = "playbook_move"
    BASIC_MOVE = "basic_move"
    SETTING = "setting"
    EXAMPLE = "example"
    ADVANCEMENT = "advancement"
    RULE = "rule"


@dataclass
class EntityReference:
    """Represents an entity mentioned in the content"""
    name: str
    entity_type: str
    confidence: float = 1.0


@dataclass
class Relationship:
    """Represents a relationship between entities"""
    source: str
    target: str
    relationship_type: str
    confidence: float = 1.0


@dataclass
class ChunkMetadata:
    """Rich metadata for each chunk, designed for graph migration"""
    chunk_id: str
    content_type: ContentType
    chapter: Optional[str] = None
    page_range: Optional[str] = None
    section_title: Optional[str] = None

    # Entity information
    entities: List[EntityReference] = None
    relationships: List[Relationship] = None

    # MASKS-specific metadata
    playbook_name: Optional[str] = None
    move_name: Optional[str] = None
    stat_used: Optional[str] = None
    archetype: Optional[str] = None

    # Cross-references
    cross_references: List[str] = None
    examples_included: bool = False

    # Graph-ready information
    node_candidates: List[Dict[str, Any]] = None
    edge_candidates: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.relationships is None:
            self.relationships = []
        if self.cross_references is None:
            self.cross_references = []
        if self.node_candidates is None:
            self.node_candidates = []
        if self.edge_candidates is None:
            self.edge_candidates = []


class MASKSContentAnalyzer:
    """Analyzes MASKS content to extract entities and relationships"""

    def __init__(self):
        # MASKS-specific patterns
        self.playbook_patterns = [
            r"THE\s+([A-Z]+)(?:\s+PLAYBOOK)?",
            r"(?:Playing\s+)?The\s+([A-Z][a-z]+)(?:\s+is|\s+feels)",
        ]

        self.move_patterns = [
            r"❑\s+([^:]+):",  # Move names with checkboxes
            r"When you ([^,]+),\s*roll",  # Move triggers
        ]

        self.label_patterns = [
            r"(DANGER|FREAK|SAVIOR|SUPERIOR|MUNDANE)",
            r"roll\s*\+\s*(Danger|Freak|Savior|Superior|Mundane)",
        ]

        self.condition_patterns = [
            r"❑\s+(Afraid|Angry|Guilty|Hopeless|Insecure)",
        ]

        # Relationship indicators
        self.relationship_patterns = [
            (r"roll\s*\+\s*(\w+)", "USES_STAT", r"(\w+)"),
            (r"When you\s+([^,]+)", "TRIGGERED_BY", r"([^,]+)"),
            (r"([A-Z][a-z]+)\s+is\s+your\s+(love|rival)", "HAS_RELATIONSHIP", r"(love|rival)"),
        ]

    def analyze_content(self, text: str, content_type: ContentType) -> ChunkMetadata:
        """Analyze content and extract structured metadata"""
        chunk_id = str(uuid.uuid4())

        entities = self._extract_entities(text)
        relationships = self._extract_relationships(text)

        # Extract MASKS-specific information
        playbook_name = self._extract_playbook_name(text)
        move_name = self._extract_move_name(text)
        stat_used = self._extract_stat_used(text)

        # Generate graph-ready data
        node_candidates = self._generate_node_candidates(entities, content_type)
        edge_candidates = self._generate_edge_candidates(relationships)

        return ChunkMetadata(
            chunk_id=chunk_id,
            content_type=content_type,
            entities=entities,
            relationships=relationships,
            playbook_name=playbook_name,
            move_name=move_name,
            stat_used=stat_used,
            cross_references=self._extract_cross_references(text),
            examples_included=self._has_examples(text),
            node_candidates=node_candidates,
            edge_candidates=edge_candidates
        )

    def _extract_entities(self, text: str) -> List[EntityReference]:
        """Extract entities from text"""
        entities = []

        # Extract playbooks
        for pattern in self.playbook_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append(EntityReference(
                    name=match.group(1).title(),
                    entity_type="Playbook",
                    confidence=0.9
                ))

        # Extract moves
        for pattern in self.move_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append(EntityReference(
                    name=match.group(1).strip(),
                    entity_type="Move",
                    confidence=0.8
                ))

        # Extract labels
        for match in re.finditer(self.label_patterns[0], text):
            entities.append(EntityReference(
                name=match.group(1),
                entity_type="Label",
                confidence=1.0
            ))

        # Extract conditions
        for match in re.finditer(self.condition_patterns[0], text):
            entities.append(EntityReference(
                name=match.group(1),
                entity_type="Condition",
                confidence=1.0
            ))

        return entities

    def _extract_relationships(self, text: str) -> List[Relationship]:
        """Extract relationships from text"""
        relationships = []

        for pattern, rel_type, target_pattern in self.relationship_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                relationships.append(Relationship(
                    source=match.group(1),
                    target="",  # Will be filled based on context
                    relationship_type=rel_type,
                    confidence=0.7
                ))

        return relationships

    def _extract_playbook_name(self, text: str) -> Optional[str]:
        """Extract playbook name if this is playbook content"""
        for pattern in self.playbook_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).title()
        return None

    def _extract_move_name(self, text: str) -> Optional[str]:
        """Extract move name if this is a move description"""
        # Look for checkbox pattern indicating a move
        match = re.search(r"❑\s+([^:]+):", text)
        if match:
            return match.group(1).strip()
        return None

    def _extract_stat_used(self, text: str) -> Optional[str]:
        """Extract which stat/label is used in a move"""
        match = re.search(r"roll\s*\+\s*(\w+)", text, re.IGNORECASE)
        if match:
            return match.group(1).title()
        return None

    def _extract_cross_references(self, text: str) -> List[str]:
        """Extract cross-references to other sections"""
        refs = []

        # Look for page references
        page_refs = re.findall(r"page\s+(\d+)", text, re.IGNORECASE)
        refs.extend([f"page_{p}" for p in page_refs])

        # Look for section references
        section_refs = re.findall(r"see\s+([A-Z][^.]+)", text)
        refs.extend(section_refs)

        return refs

    def _has_examples(self, text: str) -> bool:
        """Check if text contains examples"""
        example_indicators = [
            "for example", "Joe's character", "Marissa",
            "says Joe", "I reply", "awesome!"
        ]
        return any(indicator in text.lower() for indicator in example_indicators)

    def _generate_node_candidates(self, entities: List[EntityReference],
                                  content_type: ContentType) -> List[Dict[str, Any]]:
        """Generate potential graph nodes from entities"""
        nodes = []

        for entity in entities:
            node = {
                "id": f"{entity.entity_type}_{entity.name}".replace(" ", "_"),
                "type": entity.entity_type,
                "name": entity.name,
                "properties": {
                    "confidence": entity.confidence,
                    "source_content_type": content_type.value
                }
            }
            nodes.append(node)

        return nodes

    def _generate_edge_candidates(self, relationships: List[Relationship]) -> List[Dict[str, Any]]:
        """Generate potential graph edges from relationships"""
        edges = []

        for rel in relationships:
            edge = {
                "source": rel.source.replace(" ", "_"),
                "target": rel.target.replace(" ", "_") if rel.target else None,
                "type": rel.relationship_type,
                "properties": {
                    "confidence": rel.confidence
                }
            }
            edges.append(edge)

        return edges


class MASKSChunker:
    """Intelligent chunker for MASKS content"""

    def __init__(self, analyzer: MASKSContentAnalyzer):
        self.analyzer = analyzer

    def chunk_content(self, text: str, source_info: Dict[str, Any]) -> List[Tuple[str, ChunkMetadata]]:
        """Chunk content intelligently based on MASKS structure"""
        chunks = []

        # Try to identify content type first
        content_type = self._identify_content_type(text)

        if content_type == ContentType.PLAYBOOK:
            chunks = self._chunk_playbook(text, source_info)
        elif "CHAPTER" in text.upper() and "BASICS" in text.upper():
            chunks = self._chunk_basics_chapter(text, source_info)
        else:
            # Fallback to section-based chunking
            chunks = self._chunk_by_sections(text, source_info, content_type)

        return chunks

    def _identify_content_type(self, text: str) -> ContentType:
        """Identify what type of content this is"""
        if re.search(r"THE\s+[A-Z]+.*HERO NAME", text, re.DOTALL):
            return ContentType.PLAYBOOK
        elif re.search(r"When you.*roll.*\+", text):
            return ContentType.BASIC_MOVE
        elif "CHAPTER" in text.upper():
            return ContentType.CORE_PHILOSOPHY
        elif "Halcyon City" in text:
            return ContentType.SETTING
        else:
            return ContentType.RULE

    def _chunk_playbook(self, text: str, source_info: Dict[str, Any]) -> List[Tuple[str, ChunkMetadata]]:
        """Chunk a complete playbook"""
        chunks = []

        # Split playbook into logical sections
        sections = {
            "identity": r"(HERO NAME.*?)(?=Abilities|$)",
            "abilities": r"(Abilities.*?)(?=Moment of Truth|$)",
            "moment_of_truth": r"(Moment of Truth.*?)(?=Team Moves|$)",
            "team_moves": r"(Team Moves.*?)(?=Labels|POTENTIAL|$)",
            "mechanics": r"(Labels.*?Conditions.*?)(?=Backstory|$)",
            "backstory": r"(Backstory.*?)(?=Relationships|When our team|$)",
            "relationships": r"((?:Relationships|When our team).*?)(?=Influence|$)",
            "moves": r"((?:Beacon|Bull) Moves.*?)(?=Drives|Advancement|The Bull's Heart|$)",
            "special_systems": r"((?:Drives|The Bull's Heart).*?)$"
        }

        playbook_name = self.analyzer._extract_playbook_name(text)

        for section_name, pattern in sections.items():
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                section_text = match.group(1).strip()
                if len(section_text) > 50:  # Skip very short sections
                    metadata = self.analyzer.analyze_content(section_text, ContentType.PLAYBOOK)
                    metadata.playbook_name = playbook_name
                    metadata.section_title = f"{playbook_name} - {section_name.title()}"
                    metadata.chapter = f"Playbook: {playbook_name}"

                    chunks.append((section_text, metadata))

        return chunks

    def _chunk_basics_chapter(self, text: str, source_info: Dict[str, Any]) -> List[Tuple[str, ChunkMetadata]]:
        """Chunk the basics chapter by conceptual sections"""
        sections = [
            ("The Conversation", r"THE CONVERSATION(.*?)(?=FRAMING SCENES|$)", ContentType.CORE_PHILOSOPHY),
            ("Framing Scenes", r"FRAMING SCENES(.*?)(?=HARD FRAMING|$)", ContentType.TECHNIQUE),
            ("Hard Framing", r"HARD FRAMING(.*?)(?=FOLLOWING THE FICTION|$)", ContentType.TECHNIQUE),
            ("Following the Fiction", r"FOLLOWING THE FICTION(.*?)(?=MOVES AND DICE|$)", ContentType.RULE),
            ("Moves and Dice", r"MOVES AND DICE(.*?)(?=HITS AND MISSES|$)", ContentType.MECHANIC),
            ("Hits and Misses", r"HITS AND MISSES(.*?)(?=TRIGGERS AND UNCERTAINTY|$)", ContentType.MECHANIC),
            ("Triggers and Uncertainty", r"TRIGGERS AND UNCERTAINTY(.*?)(?=STARTING THE GAME|$)", ContentType.RULE),
            ("Starting the Game", r"STARTING THE GAME(.*?)(?=HALCYON CITY|$)", ContentType.RULE),
            ("Halcyon City", r"HALCYON CITY(.*?)(?=SETTING EXPECTATIONS|$)", ContentType.SETTING),
            ("Setting Expectations", r"SETTING EXPECTATIONS(.*?)$", ContentType.RULE),
        ]

        chunks = []
        for section_name, pattern, content_type in sections:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                section_text = match.group(1).strip()
                if len(section_text) > 100:
                    metadata = self.analyzer.analyze_content(section_text, content_type)
                    metadata.section_title = section_name
                    metadata.chapter = "Chapter 2: The Basics"
                    metadata.page_range = "27-35"  # Update based on actual pages

                    chunks.append((section_text, metadata))

        return chunks

    def _chunk_by_sections(self, text: str, source_info: Dict[str, Any],
                           content_type: ContentType) -> List[Tuple[str, ChunkMetadata]]:
        """Fallback chunking by natural breaks"""
        # Split on major headings and paragraph breaks
        sections = re.split(r'\n\s*\n(?=[A-Z][A-Z\s]+[A-Z])', text)

        chunks = []
        for i, section in enumerate(sections):
            if len(section.strip()) > 100:  # Skip very short sections
                metadata = self.analyzer.analyze_content(section, content_type)
                metadata.section_title = f"Section {i + 1}"

                chunks.append((section.strip(), metadata))

        return chunks


class MASKSRAGSystem:
    """Complete RAG system for MASKS with rich metadata"""

    def __init__(self, persist_directory: str = "./masks_rag_db"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)

        # Initialize components
        self.analyzer = MASKSContentAnalyzer()
        self.chunker = MASKSChunker(self.analyzer)

        # Initialize embedding model - using all-MiniLM-L6-v2 for good balance
        # of performance and quality for domain-specific content
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize ChromaDB with persistence
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )

        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="masks_content",
            metadata={"description": "MASKS RPG content with rich metadata"}
        )

        logger.info(f"Initialized MASKS RAG system with {self.collection.count()} existing chunks")

    def process_pdf(self, pdf_path: str) -> int:
        """Process a PDF and add it to the knowledge base"""
        logger.info(f"Processing PDF: {pdf_path}")

        # Extract text from PDF
        text = self._extract_pdf_text(pdf_path)

        # Chunk the content
        chunks = self.chunker.chunk_content(text, {"source": pdf_path})

        # Process and store chunks
        chunk_count = 0

        for chunk_text, metadata in chunks:
            logger.info(metadata)
            self._store_chunk(chunk_text, metadata)
            chunk_count += 1

        logger.info(f"Processed {chunk_count} chunks from {pdf_path}")
        return chunk_count

    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    def _store_chunk(self, text: str, metadata: ChunkMetadata):
        """Store a chunk with its rich metadata"""
        # Generate embedding
        embedding = self.embedding_model.encode(text).tolist()

        # Convert metadata to dict for storage and handle enum serialization
        metadata_dict = asdict(metadata)

        # Convert enum to string value
        if 'content_type' in metadata_dict and hasattr(metadata_dict['content_type'], 'value'):
            metadata_dict['content_type'] = metadata_dict['content_type'].value

        # Handle nested objects that ChromaDB can't store directly
        # Convert complex objects to JSON strings and handle None values
        serializable_metadata = {}

        for key, value in metadata_dict.items():
            if key in ['entities', 'relationships', 'node_candidates', 'edge_candidates']:
                # Store complex objects as JSON strings
                serializable_metadata[key] = json.dumps(value) if value else "[]"
            elif key == 'cross_references':
                # Store list as JSON string
                serializable_metadata[key] = json.dumps(value) if value else "[]"
            elif value is None:
                # Skip None values entirely as ChromaDB can't handle them reliably
                continue
            elif isinstance(value, (str, int, float, bool)):
                # Keep simple types as-is
                serializable_metadata[key] = value
            else:
                # For any other complex types, convert to string
                serializable_metadata[key] = str(value)

        logger.info(f"Storing chunk with metadata keys: {list(serializable_metadata.keys())}")

        # Store in ChromaDB
        self.collection.add(
            ids=[metadata.chunk_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[serializable_metadata]
        )

    def query(self, query_text: str, filters: Optional[Dict[str, Any]] = None,
              n_results: int = 5) -> List[Dict[str, Any]]:
        """Query the knowledge base"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query_text).tolist()

        # Build where clause from filters
        where_clause = {}
        if filters:
            for key, value in filters.items():
                if isinstance(value, list):
                    where_clause[f"{key}"] = {"$in": value}
                else:
                    where_clause[f"{key}"] = value

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause if where_clause else None
        )

        # Format results and deserialize JSON strings back to objects
        formatted_results = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i].copy()

            # Deserialize JSON strings back to Python objects
            for key in ['entities', 'relationships', 'node_candidates', 'edge_candidates', 'cross_references']:
                if key in metadata and isinstance(metadata[key], str):
                    try:
                        metadata[key] = json.loads(metadata[key])
                    except json.JSONDecodeError:
                        metadata[key] = []

            formatted_results.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': metadata,
                'distance': results['distances'][0][i]
            })

        return formatted_results

    def get_metadata_stats(self) -> Dict[str, Any]:
        """Get statistics about the stored metadata"""
        all_results = self.collection.get()

        stats = {
            'total_chunks': len(all_results['ids']),
            'content_types': {},
            'playbooks': set(),
            'chapters': set(),
            'entities_by_type': {},
        }

        for metadata in all_results['metadatas']:
            # Count content types
            content_type = metadata.get('content_type')
            stats['content_types'][content_type] = stats['content_types'].get(content_type, 0) + 1

            # Collect playbooks
            if metadata.get('playbook_name'):
                stats['playbooks'].add(metadata['playbook_name'])

            # Collect chapters
            if metadata.get('chapter'):
                stats['chapters'].add(metadata['chapter'])

            # Count entities by type - deserialize if needed
            entities_str = metadata.get('entities', '[]')
            try:
                entities = json.loads(entities_str) if isinstance(entities_str, str) else entities_str
            except json.JSONDecodeError:
                entities = []

            for entity in entities:
                entity_type = entity.get('entity_type', 'unknown')
                stats['entities_by_type'][entity_type] = stats['entities_by_type'].get(entity_type, 0) + 1

        # Convert sets to lists for JSON serialization
        stats['playbooks'] = list(stats['playbooks'])
        stats['chapters'] = list(stats['chapters'])

        return stats

    def export_graph_data(self, output_path: str):
        """Export data in format ready for graph database migration"""
        all_results = self.collection.get()

        nodes = []
        edges = []

        for metadata in all_results['metadatas']:
            # Collect nodes
            for node_candidate in metadata.get('node_candidates', []):
                nodes.append(node_candidate)

            # Collect edges
            for edge_candidate in metadata.get('edge_candidates', []):
                edges.append(edge_candidate)

        graph_data = {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'export_timestamp': str(pd.Timestamp.now()),
                'total_chunks': len(all_results['ids']),
                'source': 'MASKS RAG System'
            }
        }

        with open(output_path, 'w') as f:
            json.dump(graph_data, f, indent=2)

        logger.info(f"Exported graph data to {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize the system
    rag_system = MASKSRAGSystem()

    # Process PDFs (you would replace these with your actual PDF paths)
    for (root, dirs, files) in os.walk("./game_files"):
        for file in files:
            file = os.path.join(root, file)
            rag_system.process_pdf(file)

    # Example queries
    results = rag_system.query("What are the MC principles in masks?")
    print("Query results:", len(results))

    # Get stats
    stats = rag_system.get_metadata_stats()
    print("Database stats:", json.dumps(stats, indent=2))

    # Export for graph migration
    # rag_system.export_graph_data("masks_graph_data.json")
