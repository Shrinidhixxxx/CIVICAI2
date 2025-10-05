"""
RAG (Retrieval-Augmented Generation) Module for CivicMindAI
Handles real-time data retrieval from Chennai civic sources
"""

import requests
from bs4 import BeautifulSoup
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any, Optional
import json
import time
from datetime import datetime, timedelta
import logging

class RAGModule:
    """
    RAG Module for retrieving and processing live Chennai civic data
    """
    
    def __init__(self, civic_data: Dict[str, Any]):
        self.civic_data = civic_data
        self.sentence_model = None
        self.index = None
        self.documents = []
        self.civic_sources = self._get_civic_sources()
        self.setup_embeddings()
    
    def _get_civic_sources(self) -> List[Dict[str, str]]:
        """Get list of Chennai civic data sources"""
        return [
            {
                "name": "Greater Chennai Corporation",
                "url": "https://chennaicorporation.gov.in",
                "type": "complaints",
                "selectors": [".news-item", ".announcement", ".circular"]
            },
            {
                "name": "Chennai Metro Water",
                "url": "https://cmwssb.tn.gov.in/latest-news",
                "type": "water_supply",
                "selectors": [".news-content", ".press-release", ".notification"]
            },
            {
                "name": "TANGEDCO Chennai",
                "url": "https://www.tangedco.gov.in",
                "type": "electricity",
                "selectors": [".news-item", ".circular", ".notification"]
            }
        ]
    
    def setup_embeddings(self):
        """Setup sentence transformer and FAISS index"""
        try:
            # Use a lightweight model for faster processing
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize with basic civic information
            self._initialize_base_documents()
            
            if self.documents:
                embeddings = self.sentence_model.encode(self.documents)
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(embeddings.astype('float32'))
                
        except Exception as e:
            logging.warning(f"Could not initialize embeddings: {e}")
            self.sentence_model = None
            self.index = None
    
    def _initialize_base_documents(self):
        """Initialize with base civic documents"""
        self.documents = [
            "Greater Chennai Corporation handles garbage collection, road maintenance, and building approvals. Contact: 1913",
            "Chennai Metro Water (CMWSSB) manages water supply and sewerage. 24x7 Complaint Cell: 044-4567-4567",
            "TANGEDCO handles electricity supply and billing. Emergency: 94987-94987, Power complaints: 1912",
            "TNSTC operates bus transport services. Contact: 1800-599-1500, WhatsApp: 94450-14448",
            "Adyar area falls under Zone 13 with wards 170-182. Common issues: water supply, flooding",
            "T. Nagar falls under Zone 9-10, managed by Urbaser Sumeet for waste collection",
            "Velachery area in Zone 14-15 commonly faces power outages and water shortage",
            "Anna Nagar in Zone 6-7 has mixed waste management by GCC and Ramky",
            "Water supply complaints typically take 24 hours response time",
            "Electricity restoration should happen within 1 hour in urban areas",
            "Garbage collection is done daily door-to-door using Battery Operated Vehicles",
            "Property tax and building approvals are handled by respective zone offices",
            "Emergency water tankers can be requested during shortage periods",
            "Street lighting issues should be reported to zone offices with exact location",
            "Traffic signal problems are handled by Chennai Traffic Police coordination"
        ]
        
        # Add department-specific information
        for dept_name, dept_info in self.civic_data.get("departments_complete", {}).items():
            dept_display = dept_name.replace("_", " ")
            services = dept_info.get("services", [])
            contact = dept_info.get("main_contact", "Contact not available")
            
            doc = f"{dept_display} provides {', '.join(services)}. Contact: {contact}"
            self.documents.append(doc)
    
    def retrieve_information(self, query: str, area_info: Dict, issue_type: str) -> Dict[str, Any]:
        """
        Retrieve relevant civic information for the query
        """
        try:
            # Try to fetch live data first
            live_data = self._fetch_live_civic_data(area_info, issue_type)
            
            # Get relevant documents using similarity search
            relevant_docs = self._similarity_search(query)
            
            # Determine responsible department
            department_info = self._get_department_info(issue_type)
            
            return {
                "department": department_info.get("name", "Unknown"),
                "contact": department_info.get("contact", "N/A"),
                "services": department_info.get("services", []),
                "live_data": live_data,
                "relevant_documents": relevant_docs,
                "sources": self._get_sources(issue_type),
                "area_specific": self._get_area_specific_info(area_info, issue_type)
            }
            
        except Exception as e:
            logging.error(f"Error in retrieve_information: {e}")
            return self._get_fallback_info(issue_type)
    
    def _fetch_live_civic_data(self, area_info: Dict, issue_type: str) -> List[Dict]:
        """
        Fetch live data from civic websites (with timeout and error handling)
        """
        live_data = []
        
        for source in self.civic_sources:
            if self._is_relevant_source(source, issue_type):
                try:
                    # Add timeout and user agent
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    response = requests.get(source["url"], headers=headers, timeout=5)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Extract relevant content
                        for selector in source["selectors"]:
                            elements = soup.select(selector)
                            for element in elements[:3]:  # Limit to 3 items per source
                                text = element.get_text(strip=True)
                                if len(text) > 50:  # Only meaningful content
                                    live_data.append({
                                        "source": source["name"],
                                        "content": text[:200],  # Limit content length
                                        "type": source["type"],
                                        "timestamp": datetime.now().isoformat()
                                    })
                        
                        if live_data:
                            break  # Stop after getting data from first successful source
                            
                except Exception as e:
                    logging.warning(f"Could not fetch from {source['name']}: {e}")
                    continue
        
        return live_data
    
    def _is_relevant_source(self, source: Dict, issue_type: str) -> bool:
        """Check if source is relevant for the issue type"""
        relevance_map = {
            "water_supply": ["water_supply"],
            "electricity": ["electricity"], 
            "garbage_collection": ["complaints"],
            "roads": ["complaints"],
            "transport": ["transport", "complaints"]
        }
        
        return source["type"] in relevance_map.get(issue_type, ["complaints"])
    
    def _similarity_search(self, query: str, k: int = 3) -> List[str]:
        """
        Perform similarity search on documents
        """
        if not self.sentence_model or not self.index:
            # Fallback to keyword matching
            return self._keyword_search(query)
        
        try:
            query_embedding = self.sentence_model.encode([query])
            _, indices = self.index.search(query_embedding.astype('float32'), k)
            
            relevant_docs = []
            for idx in indices[0]:
                if 0 <= idx < len(self.documents):
                    relevant_docs.append(self.documents[idx])
            
            return relevant_docs
            
        except Exception as e:
            logging.warning(f"Similarity search failed: {e}")
            return self._keyword_search(query)
    
    def _keyword_search(self, query: str) -> List[str]:
        """Fallback keyword-based search"""
        query_lower = query.lower()
        relevant_docs = []
        
        for doc in self.documents:
            doc_lower = doc.lower()
            # Simple keyword matching
            if any(word in doc_lower for word in query_lower.split() if len(word) > 3):
                relevant_docs.append(doc)
                if len(relevant_docs) >= 3:
                    break
        
        return relevant_docs or self.documents[:3]  # Return first 3 if no matches
    
    def _get_department_info(self, issue_type: str) -> Dict[str, Any]:
        """Get department information based on issue type"""
        department_mapping = {
            "water_supply": "Chennai_Metro_Water",
            "electricity": "TANGEDCO",
            "garbage_collection": "Greater_Chennai_Corporation",
            "roads": "Greater_Chennai_Corporation",
            "transport": "TNSTC"
        }
        
        dept_key = department_mapping.get(issue_type, "Greater_Chennai_Corporation")
        dept_data = self.civic_data.get("departments_complete", {}).get(dept_key, {})
        
        return {
            "name": dept_key.replace("_", " "),
            "contact": dept_data.get("main_contact", "1913"),
            "services": dept_data.get("services", [])
        }
    
    def _get_sources(self, issue_type: str) -> List[str]:
        """Get sources for the information"""
        return [
            f"Live data from Chennai civic portals",
            f"Official department websites",
            f"Government notification systems"
        ]
    
    def _get_area_specific_info(self, area_info: Dict, issue_type: str) -> Dict[str, Any]:
        """Get area-specific information"""
        area_name = area_info.get("name", "Chennai")
        
        # Check if area has specific information in civic data
        area_specific = {}
        
        if area_info.get("type") == "zone":
            zone_data = self.civic_data.get("zones_complete", {}).get(area_info.get("zone_number", ""), {})
            area_specific = {
                "zone_info": zone_data,
                "ward_count": len(zone_data.get("wards", [])),
                "assembly_constituency": zone_data.get("assembly_constituency", "N/A")
            }
        
        return area_specific
    
    def _get_fallback_info(self, issue_type: str) -> Dict[str, Any]:
        """Fallback information when retrieval fails"""
        fallback_info = {
            "water_supply": {
                "department": "Chennai Metro Water",
                "contact": "044-4567-4567",
                "services": ["Water supply", "Sewerage", "New connections"]
            },
            "electricity": {
                "department": "TANGEDCO", 
                "contact": "94987-94987",
                "services": ["Power supply", "Billing", "Complaints"]
            },
            "garbage_collection": {
                "department": "Greater Chennai Corporation",
                "contact": "1913",
                "services": ["Waste collection", "Street cleaning"]
            }
        }
        
        info = fallback_info.get(issue_type, fallback_info["garbage_collection"])
        info.update({
            "live_data": [],
            "relevant_documents": [f"Standard information for {issue_type} issues"],
            "sources": ["Fallback civic information"],
            "area_specific": {}
        })
        
        return info
    
    def update_documents(self, new_documents: List[str]):
        """Update document index with new information"""
        if new_documents and self.sentence_model:
            try:
                self.documents.extend(new_documents)
                
                # Recreate index with all documents
                embeddings = self.sentence_model.encode(self.documents)
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                self.index.add(embeddings.astype('float32'))
                
            except Exception as e:
                logging.error(f"Error updating documents: {e}")
    
    def get_cache_key(self, query: str, area_info: Dict, issue_type: str) -> str:
        """Generate cache key for query"""
        key_parts = [
            query.lower().strip(),
            area_info.get("name", ""),
            issue_type
        ]
        return "_".join(key_parts).replace(" ", "_")