"""
KAG (Knowledge Graph Augmented Generation) Module for CivicMindAI
Handles graph-based reasoning over Chennai civic entities
"""

import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
import json
import logging
from datetime import datetime

class KnowledgeGraphModule:
    """
    Knowledge Graph Module for reasoning over Chennai civic entities and relationships
    """
    
    def __init__(self, civic_data: Dict[str, Any]):
        self.civic_data = civic_data
        self.graph = nx.DiGraph()
        self.build_knowledge_graph()
    
    def build_knowledge_graph(self):
        """Build the Chennai civic knowledge graph"""
        try:
            self._add_administrative_entities()
            self._add_department_entities()
            self._add_service_entities()
            self._add_area_entities()
            self._add_issue_entities()
            self._add_relationships()
            
            logging.info(f"Knowledge graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            
        except Exception as e:
            logging.error(f"Error building knowledge graph: {e}")
            self._build_minimal_graph()
    
    def _add_administrative_entities(self):
        """Add Chennai administrative structure to graph"""
        # Add Chennai as root entity
        self.graph.add_node("Chennai", 
                          type="city", 
                          population="7.9M", 
                          zones=15, 
                          wards=200)
        
        # Add zones
        zones_data = self.civic_data.get("zones_complete", {})
        for zone_key, zone_info in zones_data.items():
            zone_name = zone_key.replace("_", " ")
            self.graph.add_node(zone_name,
                              type="zone",
                              wards=zone_info.get("wards", []),
                              ward_names=zone_info.get("ward_names", []),
                              assembly_constituency=zone_info.get("assembly_constituency", ""),
                              parliament_constituency=zone_info.get("parliament_constituency", ""))
            
            # Connect zone to Chennai
            self.graph.add_edge("Chennai", zone_name, relationship="contains")
            
            # Add wards
            for i, ward_num in enumerate(zone_info.get("wards", [])):
                ward_name = f"Ward_{ward_num}"
                ward_display_name = zone_info.get("ward_names", [])
                if i < len(ward_display_name):
                    ward_display = ward_display_name[i]
                else:
                    ward_display = f"Ward {ward_num}"
                
                self.graph.add_node(ward_name,
                                  type="ward",
                                  number=ward_num,
                                  display_name=ward_display,
                                  zone=zone_name)
                
                # Connect ward to zone
                self.graph.add_edge(zone_name, ward_name, relationship="contains")
    
    def _add_department_entities(self):
        """Add civic departments to graph"""
        departments_data = self.civic_data.get("departments_complete", {})
        
        for dept_key, dept_info in departments_data.items():
            dept_name = dept_key.replace("_", " ")
            
            self.graph.add_node(dept_name,
                              type="department",
                              main_contact=dept_info.get("main_contact", ""),
                              website=dept_info.get("website", ""),
                              services=dept_info.get("services", []))
            
            # Connect department to Chennai
            self.graph.add_edge("Chennai", dept_name, relationship="administered_by")
            
            # Add zone offices if available
            zone_offices = dept_info.get("zone_offices", {})
            for zone, office_info in zone_offices.items():
                office_name = f"{dept_name} {zone} Office"
                self.graph.add_node(office_name,
                                  type="zone_office",
                                  contact=office_info.get("contact", ""),
                                  location=office_info.get("location", ""))
                
                # Connect office to department and zone
                self.graph.add_edge(dept_name, office_name, relationship="has_office")
                zone_display = zone.replace("_", " ")
                if zone_display in self.graph:
                    self.graph.add_edge(zone_display, office_name, relationship="served_by")
    
    def _add_service_entities(self):
        """Add civic services to graph"""
        service_categories = {
            "Water Services": ["Water supply", "Sewerage", "Water tax", "New connections", "Tanker services"],
            "Waste Management": ["Garbage collection", "Street cleaning", "Waste disposal", "Recycling"],
            "Electricity Services": ["Power supply", "Billing", "New connections", "Meter reading", "Fault repair"],
            "Transport Services": ["Bus transport", "Route planning", "Fare collection", "Driver services"],
            "Infrastructure": ["Road maintenance", "Street lighting", "Traffic signals", "Drainage"],
            "Administrative": ["Birth/Death certificates", "Property tax", "Building approvals", "Licenses"]
        }
        
        for category, services in service_categories.items():
            self.graph.add_node(category, type="service_category")
            self.graph.add_edge("Chennai", category, relationship="provides")
            
            for service in services:
                self.graph.add_node(service, type="service", category=category)
                self.graph.add_edge(category, service, relationship="includes")
                
                # Connect services to appropriate departments
                self._connect_services_to_departments(service)
    
    def _connect_services_to_departments(self, service: str):
        """Connect services to responsible departments"""
        service_dept_mapping = {
            "Water supply": "Chennai Metro Water",
            "Sewerage": "Chennai Metro Water", 
            "Water tax": "Chennai Metro Water",
            "Garbage collection": "Greater Chennai Corporation",
            "Road maintenance": "Greater Chennai Corporation",
            "Street lighting": "Greater Chennai Corporation",
            "Property tax": "Greater Chennai Corporation",
            "Building approvals": "Greater Chennai Corporation",
            "Power supply": "TANGEDCO",
            "Billing": "TANGEDCO",
            "Bus transport": "TNSTC",
            "Route planning": "TNSTC"
        }
        
        dept = service_dept_mapping.get(service)
        if dept and dept in self.graph:
            self.graph.add_edge(dept, service, relationship="provides")
    
    def _add_area_entities(self):
        """Add area-specific entities"""
        # Add pincode areas
        for pincode, area_name in self.civic_data.get("pincode_mapping", {}).items():
            area_node = f"Area_{area_name.replace(' ', '_')}"
            self.graph.add_node(area_node,
                              type="area",
                              name=area_name,
                              pincode=pincode)
            
            self.graph.add_edge("Chennai", area_node, relationship="contains")
    
    def _add_issue_entities(self):
        """Add common civic issues to graph"""
        issues_data = self.civic_data.get("common_issues_database", {})
        
        for issue_key, issue_info in issues_data.items():
            issue_name = issue_key.replace("_", " ").title()
            
            self.graph.add_node(issue_name,
                              type="issue_type",
                              keywords=issue_info.get("keywords", []),
                              department=issue_info.get("department", ""),
                              response_time=issue_info.get("typical_response_time", ""))
            
            # Connect to relevant department
            dept = issue_info.get("department", "")
            if dept in self.graph:
                self.graph.add_edge(dept, issue_name, relationship="handles")
            
            self.graph.add_edge("Chennai", issue_name, relationship="experiences")
    
    def _add_relationships(self):
        """Add additional relationships between entities"""
        # Add escalation relationships
        escalation_chains = {
            "Chennai Metro Water": ["Area Engineer", "Executive Engineer", "Superintending Engineer", "Chief Engineer"],
            "TANGEDCO": ["Section Officer", "Sub Divisional Officer", "Divisional Officer", "Superintending Engineer"],
            "Greater Chennai Corporation": ["Zone Officer", "Assistant Commissioner", "Deputy Commissioner", "Commissioner"]
        }
        
        for dept, chain in escalation_chains.items():
            if dept in self.graph:
                for i, position in enumerate(chain):
                    position_node = f"{dept}_{position}".replace(" ", "_")
                    self.graph.add_node(position_node,
                                      type="position",
                                      department=dept,
                                      level=i+1)
                    
                    self.graph.add_edge(dept, position_node, relationship="has_position")
                    
                    # Add escalation relationships
                    if i > 0:
                        prev_position = f"{dept}_{chain[i-1]}".replace(" ", "_")
                        self.graph.add_edge(position_node, prev_position, relationship="escalates_to")
    
    def _build_minimal_graph(self):
        """Build minimal graph if main build fails"""
        self.graph.add_node("Chennai", type="city")
        
        basic_departments = ["Greater Chennai Corporation", "Chennai Metro Water", "TANGEDCO", "TNSTC"]
        for dept in basic_departments:
            self.graph.add_node(dept, type="department")
            self.graph.add_edge("Chennai", dept, relationship="administered_by")
    
    def reason_over_graph(self, query: str, area_info: Dict, issue_type: str) -> Dict[str, Any]:
        """
        Apply graph reasoning to provide contextual information
        """
        try:
            reasoning_results = {
                "path_analysis": self._analyze_paths(area_info, issue_type),
                "related_entities": self._find_related_entities(issue_type),
                "escalation_path": self._get_escalation_path(issue_type),
                "area_connections": self._get_area_connections(area_info),
                "service_dependencies": self._get_service_dependencies(issue_type),
                "recommendations": self._generate_recommendations(area_info, issue_type)
            }
            
            return reasoning_results
            
        except Exception as e:
            logging.error(f"Error in graph reasoning: {e}")
            return self._get_fallback_reasoning(area_info, issue_type)
    
    def _analyze_paths(self, area_info: Dict, issue_type: str) -> List[List[str]]:
        """Analyze paths between area and responsible entities"""
        paths = []
        
        try:
            area_name = area_info.get("name", "Chennai")
            issue_name = issue_type.replace("_", " ").title()
            
            # Find paths from area to issue handling
            area_nodes = [n for n in self.graph.nodes() if area_name.lower() in n.lower()]
            issue_nodes = [n for n in self.graph.nodes() if issue_name.lower() in n.lower()]
            
            for area_node in area_nodes[:2]:  # Limit to 2 area nodes
                for issue_node in issue_nodes[:2]:  # Limit to 2 issue nodes
                    try:
                        path = nx.shortest_path(self.graph, area_node, issue_node)
                        if len(path) <= 5:  # Only reasonable length paths
                            paths.append(path)
                    except nx.NetworkXNoPath:
                        continue
            
        except Exception as e:
            logging.warning(f"Error analyzing paths: {e}")
        
        return paths[:3]  # Return top 3 paths
    
    def _find_related_entities(self, issue_type: str) -> List[Dict[str, Any]]:
        """Find entities related to the issue type"""
        related = []
        
        try:
            issue_name = issue_type.replace("_", " ").title()
            
            # Find direct neighbors of issue-related nodes
            issue_nodes = [n for n in self.graph.nodes() if issue_name.lower() in n.lower()]
            
            for issue_node in issue_nodes:
                neighbors = list(self.graph.neighbors(issue_node))
                for neighbor in neighbors[:5]:  # Limit neighbors
                    node_data = self.graph.nodes[neighbor]
                    related.append({
                        "entity": neighbor,
                        "type": node_data.get("type", "unknown"),
                        "relationship": "related_to_issue"
                    })
            
        except Exception as e:
            logging.warning(f"Error finding related entities: {e}")
        
        return related[:10]  # Return top 10 related entities
    
    def _get_escalation_path(self, issue_type: str) -> List[str]:
        """Get escalation path for the issue type"""
        try:
            # Map issue types to departments
            dept_mapping = {
                "water_supply": "Chennai Metro Water",
                "electricity": "TANGEDCO", 
                "garbage_collection": "Greater Chennai Corporation",
                "roads": "Greater Chennai Corporation",
                "transport": "TNSTC"
            }
            
            dept = dept_mapping.get(issue_type, "Greater Chennai Corporation")
            
            # Find escalation chain
            escalation_nodes = [n for n in self.graph.nodes() 
                              if self.graph.nodes[n].get("department") == dept 
                              and self.graph.nodes[n].get("type") == "position"]
            
            # Sort by level
            escalation_nodes.sort(key=lambda x: self.graph.nodes[x].get("level", 0))
            
            return [node.split("_")[-1].replace("_", " ") for node in escalation_nodes]
            
        except Exception as e:
            logging.warning(f"Error getting escalation path: {e}")
            return ["First Level Officer", "Senior Officer", "Department Head"]
    
    def _get_area_connections(self, area_info: Dict) -> Dict[str, Any]:
        """Get connections specific to the area"""
        connections = {}
        
        try:
            area_name = area_info.get("name", "")
            
            # Find zone information
            if area_info.get("type") == "zone":
                zone_name = area_info.get("zone_number", "").replace("_", " ")
                if zone_name in self.graph:
                    node_data = self.graph.nodes[zone_name]
                    connections["zone_info"] = {
                        "assembly_constituency": node_data.get("assembly_constituency", ""),
                        "parliament_constituency": node_data.get("parliament_constituency", ""),
                        "ward_count": len(node_data.get("wards", []))
                    }
            
            # Find area-specific services
            area_nodes = [n for n in self.graph.nodes() if area_name.lower() in n.lower()]
            for area_node in area_nodes:
                neighbors = list(self.graph.neighbors(area_node))
                connections["nearby_services"] = [n for n in neighbors 
                                                if self.graph.nodes[n].get("type") == "service"][:5]
            
        except Exception as e:
            logging.warning(f"Error getting area connections: {e}")
        
        return connections
    
    def _get_service_dependencies(self, issue_type: str) -> List[str]:
        """Get dependencies for the service/issue"""
        dependencies = []
        
        try:
            issue_name = issue_type.replace("_", " ").title()
            
            # Find service dependencies through graph traversal
            service_nodes = [n for n in self.graph.nodes() 
                           if self.graph.nodes[n].get("type") == "service" 
                           and issue_name.lower() in n.lower()]
            
            for service_node in service_nodes:
                # Find what this service depends on
                predecessors = list(self.graph.predecessors(service_node))
                for pred in predecessors:
                    if self.graph.nodes[pred].get("type") in ["department", "service_category"]:
                        dependencies.append(pred)
            
        except Exception as e:
            logging.warning(f"Error getting service dependencies: {e}")
        
        return dependencies[:5]
    
    def _generate_recommendations(self, area_info: Dict, issue_type: str) -> List[str]:
        """Generate recommendations based on graph analysis"""
        recommendations = []
        
        try:
            # Basic recommendations based on issue type
            issue_recommendations = {
                "water_supply": [
                    "Contact Chennai Metro Water at 044-4567-4567",
                    "Check if area has scheduled maintenance",
                    "Request tanker service if emergency",
                    "Escalate to Superintending Engineer if no response"
                ],
                "electricity": [
                    "Call TANGEDCO at 94987-94987 for power failures", 
                    "Report via WhatsApp at 94458508111 with photo",
                    "Check estimated restoration time",
                    "Claim compensation for delays over 1 hour"
                ],
                "garbage_collection": [
                    "Contact GCC at 1913 or use Namma Chennai app",
                    "Ensure waste segregation at source",
                    "Note down complaint number",
                    "Follow up with zone office if needed"
                ]
            }
            
            recommendations.extend(issue_recommendations.get(issue_type, [
                "Contact relevant department for assistance",
                "Provide exact location and issue description",
                "Keep complaint number for tracking"
            ]))
            
            # Area-specific recommendations
            area_name = area_info.get("name", "")
            if area_name:
                recommendations.append(f"Mention {area_name} location clearly when reporting")
                
                if area_info.get("type") == "zone":
                    recommendations.append(f"Contact zone office for faster resolution")
            
        except Exception as e:
            logging.warning(f"Error generating recommendations: {e}")
        
        return recommendations[:6]  # Limit to 6 recommendations
    
    def _get_fallback_reasoning(self, area_info: Dict, issue_type: str) -> Dict[str, Any]:
        """Fallback reasoning when graph analysis fails"""
        return {
            "path_analysis": [],
            "related_entities": [
                {"entity": "Greater Chennai Corporation", "type": "department", "relationship": "primary_contact"}
            ],
            "escalation_path": ["Junior Officer", "Senior Officer", "Department Head"],
            "area_connections": {"zone_info": {"note": "Area information available on request"}},
            "service_dependencies": ["Primary Department Contact"],
            "recommendations": [
                f"Contact appropriate department for {issue_type} issues",
                "Provide detailed location information",
                "Keep complaint reference number",
                "Follow up if no response within expected timeframe"
            ]
        }
    
    def add_dynamic_entity(self, entity_name: str, entity_type: str, attributes: Dict[str, Any]):
        """Add new entity to the knowledge graph dynamically"""
        try:
            self.graph.add_node(entity_name, type=entity_type, **attributes)
            logging.info(f"Added dynamic entity: {entity_name}")
        except Exception as e:
            logging.error(f"Error adding dynamic entity: {e}")
    
    def get_graph_stats(self) -> Dict[str, int]:
        """Get knowledge graph statistics"""
        try:
            return {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
                "node_types": len(set(self.graph.nodes[n].get("type", "unknown") for n in self.graph.nodes())),
                "max_degree": max(dict(self.graph.degree()).values()) if self.graph.nodes() else 0
            }
        except Exception as e:
            logging.error(f"Error getting graph stats: {e}")
            return {"total_nodes": 0, "total_edges": 0, "node_types": 0, "max_degree": 0}
    
    def export_graph(self, format_type: str = "json") -> Any:
        """Export knowledge graph in specified format"""
        try:
            if format_type == "json":
                return nx.node_link_data(self.graph)
            elif format_type == "gexf":
                return nx.write_gexf(self.graph)
            else:
                return None
        except Exception as e:
            logging.error(f"Error exporting graph: {e}")
            return None