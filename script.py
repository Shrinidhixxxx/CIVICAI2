# Create comprehensive Chennai civic data structure for CivicMindAI
import json
import pandas as pd

# Complete Chennai ward and zone data based on search results
chennai_complete_data = {
    "administrative_structure": {
        "total_zones": 15,
        "total_wards": 200,
        "expansion_plan": "20 zones by 2024-25",
        "total_population": "79.53 lakh (2023 projection)",
        "total_properties": "13.82 lakh"
    },
    
    "zones_complete": {
        "Zone_1_Thiruvottiyur": {
            "wards": list(range(1, 15)),  # Wards 1-14
            "ward_names": [
                "Kathivakkam", "Ennore", "Ernavoor", "Ajax", "Tiruvottiyur", 
                "Kaladipet", "Rajakadai", "Kodungaiyur (West)", "Kodungaiyur (East)",
                "Dr. Radhakrishnan Nagar (North)", "Cheriyan Nagar (North)", 
                "Jeeva Nagar (North)", "Cheriyan Nagar (South)", "Jeeva Nagar (South)"
            ],
            "assembly_constituency": "Thiruvottiyur",
            "parliament_constituency": "Chennai North"
        },
        
        "Zone_2_Manali": {
            "wards": list(range(15, 22)),  # Wards 15-21
            "ward_names": [
                "Edyanchavadi", "Kadapakkam", "Theeyambakkam", "Manali", 
                "Mathur", "Sanjeevirayanpet", "Grace Garden"
            ],
            "assembly_constituency": "Thiruvottiyur/Madhavaram/Ponneri",
            "parliament_constituency": "Chennai North/Thiruvallur"
        },
        
        "Zone_3_Madhavaram": {
            "wards": list(range(22, 34)),  # Wards 22-33
            "ward_names": [
                "Kavankarai", "Puzhal", "Puthagram", "Kathirvedu", 
                "Lakshmipuram – Madhavaram", "Assisi Nagar 9th St", "Chinnasekkadu",
                "Madhavaram", "Ma-Po-Si Nagar", "Royapuram", "Singarathottam", "Narayanappa Thottam"
            ],
            "assembly_constituency": "Madhavaram",
            "parliament_constituency": "Thiruvallur"
        },
        
        "Zone_4_Tondiarpet": {
            "wards": list(range(34, 49)),  # Wards 34-48
            "ward_names": [
                "Korukkupet", "Mottai Thottam", "Kumarasamy Nagar (South)", 
                "Dr. Radhakrishnan Nagar (South)", "Kumarasamy Nagar (North)",
                "Vijayaragavalu Nagar (West)", "Tondiarpet", "Old Washermenpet",
                "Meenakshiammanpet", "Kondithope", "Sevenwells (North)", "Amman Koil",
                "Muthialpet", "Vallalseethakathi Nagar", "Kachaleeswarar Nagar"
            ],
            "assembly_constituency": "Perambur/RK Nagar/Royapuram",
            "parliament_constituency": "Chennai North"
        },
        
        "Zone_5_Royapuram": {
            "wards": list(range(49, 64)),  # Wards 49-63
            "ward_names": [
                "Sevenwells (South)", "Sowcarpet", "Basin Bridge", "Vyasarpet (South)",
                "Vyasarpet (North)", "Perambur (North)", "Perambur (East)", "Elango Nagar",
                "Perambur (South)", "Thiru-Vi-Ka Nagar", "Wadia Nagar", "Dr.Sathyavanimuthu Nagar",
                "Pulianthope", "Dr.Besant Nagar", "Pedhunayakanpet"
            ],
            "assembly_constituency": "Perambur/Royapuram",
            "parliament_constituency": "Chennai North"
        },
        
        "Zone_6_Thiru_Vi_Ka_Nagar": {
            "wards": list(range(64, 79)),  # Wards 64-78
            "ward_names": [
                "Perumal Koil Thottam", "Thattankulam", "Choolai", "Poonga Nagar",
                "Elephant Gate", "Edapalayam", "Agaram (North)", "Sembiam",
                "Siruvalloor", "Nagammai Ammaiyar Nagar", "Agaram (South)",
                "Vidhudalai Gurusami Nagar", "Ayanavaram", "Nagammaiammaiyar Nagar (South)",
                "Panneer Selvam Nagar"
            ],
            "assembly_constituency": "Thiru-Vi-Ka Nagar/Dr. Radhakrishnan Nagar",
            "parliament_constituency": "Chennai Central"
        },
        
        "Zone_7_Ambattur": {
            "wards": list(range(79, 94)),  # Wards 79-93
            "ward_names": [
                "Maraimalai Adigal Nagar", "Maraimalai Adigal Nagar (South)", "Purasawalkam",
                "Kolathur", "Villiwakkam (North)", "Villiwakkam (South)", "Virugambakkam (North)",
                "Anna Nagar (West)", "Anna Nagar (Central)", "Anna Nagar (East)",
                "Shenoy Nagar", "Kilpauk (North)", "Gangadeeswarar Koil", "Kilpauk (South)",
                "Amanjikarai (North)"
            ],
            "assembly_constituency": "Ambattur/Villivakkam",
            "parliament_constituency": "Chennai Central"
        },
        
        "Zone_8_Anna_Nagar": {
            "wards": list(range(94, 109)),  # Wards 94-108
            "ward_names": [
                "Amanjikarai (Central)", "Amanjikarai (West)", "Periyar Nagar (North)",
                "Periyar Nagar (West)", "Nungambakkam", "Adikesavapuram", "Nehru Nagar",
                "Chintadripet", "Komaleeswaranpet", "Balasubramanya Nagar", "Thiruvotteeswaranpet",
                "Natesan Nagar", "Chepauk", "Zambazaar", "Umaru Pulavar Nagar"
            ],
            "assembly_constituency": "Villivakkam/Egmore/Anna Nagar",
            "parliament_constituency": "Chennai Central"
        },
        
        "Zone_9_Teynampet": {
            "wards": list(range(109, 127)),  # Wards 109-126
            "ward_names": [
                "Kannappar Nagar W(G)", "Pattalam", "Chetpet", "Egmore", "Pudupet",
                "Ko-Su-Mani Nagar", "Nakeerar Nagar", "Thousand Lights", "Azhagiri Nagar",
                "Amir Mahal", "Royapettah", "Teynampet", "Sathyamurthy Nagar",
                "Alwarpet (North)", "Alwarpet (South)", "Vadapalani (West)", "Vadapalani (East)",
                "Kalaivanar Nagar"
            ],
            "assembly_constituency": "Thousand Lights/Chepauk/Teynampet",
            "parliament_constituency": "Chennai Central"
        },
        
        "Zone_10_Kodambakkam": {
            "wards": list(range(127, 143)),  # Wards 127-142
            "ward_names": [
                "Navalar Nedunchezian Nagar (West)", "Navalar Nedunchezian Nagar (East)",
                "Ashok Nagar", "M.G.R. Nagar", "Kamaraj Nagar (North)", "Kamaraj Nagar (South)",
                "Thyagaraya Nagar", "Rajaji Nagar", "Virugambakkam (South)", "Saligramam",
                "Kodambakkam (North)", "Kodambakkam (South)", "Saidapet", "Kumaran Nagar (North)",
                "Kumaran Nagar (South)", "Saidapet (West)"
            ],
            "assembly_constituency": "T. Nagar/Kodambakkam",
            "parliament_constituency": "Chennai Central"
        },
        
        "Zone_11_Valasaravakkam": {
            "wards": list(range(143, 156)),  # Wards 143-155
            "ward_names": [
                "Kalaingar Karunanidhi Nagar", "V O C Nagar", "G D Naidu Nagar (East)",
                "G. D Naidu Nagar (West)", "Guindy (West)", "Guindy (East)", "Beemannapettai",
                "Thiruvalluvar Nagar", "Madavaperumal Puram", "Karaneeswarpuram", "Santhome",
                "Mylapore", "Avvai Nagar (South)"
            ],
            "assembly_constituency": "Valasaravakkam/Saidapet",
            "parliament_constituency": "Chennai South"
        },
        
        "Zone_12_Alandur": {
            "wards": list(range(156, 168)),  # Wards 156-167
            "ward_names": [
                "Raja Annamalai Puram", "Avvai Nagar (North)", "Adyar (West)", "Adyar (East)",
                "Velachery", "Thiruvanmiyur (West)", "Thiruvanmiyur (East)", "Besant Nagar",
                "Urur", "Adampakkam", "Velachery West", "Gandhi Salai"
            ],
            "assembly_constituency": "Alandur/Saidapet",
            "parliament_constituency": "Chennai South"
        },
        
        "Zone_13_Adyar": {
            "wards": list(range(170, 183)),  # Wards 170-182
            "ward_names": [
                "Avvai Nagar (South)", "Raja Annamalai Puram", "Avvai Nagar (North)",
                "Adyar (West)", "Adyar (East)", "Velachery", "Thiruvanmiyur (West)",
                "Thiruvanmiyur (East)", "Besant Nagar", "Urur", "Adampakkam",
                "Velachery West", "Gandhi Salai"
            ],
            "assembly_constituency": "Saidapet/Mylapore/Velachery",
            "parliament_constituency": "Chennai South"
        },
        
        "Zone_14_Perungudi": {
            "wards": [168, 169] + list(range(183, 192)),  # Wards 168, 169, 183-191
            "ward_names": [
                "Taramani", "Ullagaram", "Puzhuthivakkam", "Kottivakkam", "Pallikaranai",
                "Palavakkam", "Madipakkam", "Jaladianpet", "Neelangarai", "Thoraipakkam", "Injambakkam"
            ],
            "assembly_constituency": "Velachery/Alandur/Sholinganallur",
            "parliament_constituency": "Chennai South"
        },
        
        "Zone_15_Sholinganallur": {
            "wards": list(range(192, 201)),  # Wards 192-200
            "ward_names": [
                "Karapakkam", "Sholinganallur", "Uthandi", "Semmancheri", "Navalur",
                "Siruseri", "Kelambakkam", "Sithalapakkam", "Medavakkam"
            ],
            "assembly_constituency": "Sholinganallur",
            "parliament_constituency": "Chennai South"
        }
    },
    
    "departments_complete": {
        "Greater_Chennai_Corporation": {
            "main_contact": "1913",
            "email": "commissioner@chennaicorporation.gov.in",
            "website": "chennaicorporation.gov.in",
            "complaint_portal": "erp.chennaicorporation.gov.in/pgr/",
            "mobile_app": "Namma Chennai App",
            "services": [
                "Birth/Death certificates", "Property tax", "Building approvals", 
                "Garbage collection", "Road maintenance", "Street lighting",
                "Water drainage", "Public health", "Markets", "Parks"
            ],
            "zone_offices": {
                "Zone_1": {"contact": "044-25992828", "location": "Tiruvottiyur"},
                "Zone_2": {"contact": "044-26375560", "location": "Manali"},
                "Zone_3": {"contact": "044-26171451", "location": "Madhavaram"},
                "Zone_4": {"contact": "044-25914849", "location": "Tondiarpet"},
                "Zone_5": {"contact": "044-25953285", "location": "Royapuram"},
                "Zone_6": {"contact": "044-24753265", "location": "Thiru-Vi-Ka Nagar"},
                "Zone_7": {"contact": "044-26570570", "location": "Ambattur"},
                "Zone_8": {"contact": "044-26215969", "location": "Anna Nagar"},
                "Zone_9": {"contact": "044-28340276", "location": "Teynampet"},
                "Zone_10": {"contact": "044-24894466", "location": "Kodambakkam"},
                "Zone_11": {"contact": "044-24610405", "location": "Valasaravakkam"},
                "Zone_12": {"contact": "044-24502575", "location": "Alandur"},
                "Zone_13": {"contact": "044-24516464", "location": "Adyar"},
                "Zone_14": {"contact": "044-24503939", "location": "Perungudi"},
                "Zone_15": {"contact": "044-24502575", "location": "Sholinganallur"}
            }
        },
        
        "Chennai_Metro_Water": {
            "main_contact": "044-4567-4567",
            "complaint_cell": "044-4567-4567 (24x7)",
            "website": "cmwssb.tn.gov.in",
            "email": "cmwssb@tn.gov.in",
            "head_office": "No.1, Pumping Station Road, Chintadripet, Chennai-02",
            "services": [
                "Water supply", "Sewerage", "Water tax", "New connections", 
                "Complaints", "Tanker services", "Water quality testing"
            ],
            "area_divisions": {
                "North": {"zones": [1, 2, 3], "contact": "044-28451300 Extn.233", "se_mobile": "8144931000"},
                "North_East": {"zones": [4, 5, 6], "contact": "044-28451300 Extn.213", "se_mobile": "8144945000"},
                "Central": {"zones": [7, 8, 9], "contact": "044-28451300 Extn.212", "se_mobile": "8144934000"},
                "South_West": {"zones": [10, 11, 12], "contact": "044-28451300 Extn.386", "se_mobile": "8144930999"},
                "South": {"zones": [13, 14, 15], "contact": "044-28451300 Extn.211", "se_mobile": "8144923000"}
            }
        },
        
        "TANGEDCO": {
            "main_contact": "94987-94987",
            "emergency_contact": "1912",
            "website": "tangedco.gov.in",
            "complaint_portal": "www.tangedco.tn.gov.in",
            "whatsapp_complaints": "94458508111",
            "services": [
                "Electricity supply", "Power failures", "Billing complaints",
                "New connections", "Meter issues", "Safety complaints"
            ],
            "compensation_policy": "Rs. 50 per 6 hours delay for power restoration",
            "response_time_urban": "1 hour",
            "circle_offices": {
                "Chennai_North": {
                    "areas": ["Thiruvottiyur", "Manali", "Madhavaram", "Tondiarpet", "Royapuram", "Anna Nagar", "Ambattur"],
                    "contact": "044-26214427",
                    "foc_centers": [
                        {"area": "Arumbakkam", "contact": "044-23631714", "mobile": "9445850381"},
                        {"area": "Anna Nagar", "contact": "044-26214427", "mobile": "9445850381"},
                        {"area": "Ambattur", "contact": "044-26242004", "mobile": "9445850392"},
                        {"area": "Avadi", "contact": "044-26384010", "mobile": "9445850397"}
                    ]
                },
                "Chennai_Central": {
                    "areas": ["Teynampet", "Kodambakkam", "Valasaravakkam"],
                    "contact": "044-28340276",
                    "foc_centers": [
                        {"area": "Chetpet", "contact": "044-26414398", "mobile": "9445850378"},
                        {"area": "Koyambedu", "contact": "044-24799171", "mobile": "9445850383"},
                        {"area": "Shenoy Nagar", "contact": "044-26214427", "mobile": "9445850380"}
                    ]
                },
                "Chennai_South": {
                    "areas": ["Alandur", "Adyar", "Perungudi", "Sholinganallur"],
                    "contact": "044-24913001",
                    "foc_centers": [
                        {"area": "Velachery", "contact": "044-22450001", "mobile": "9445850439"},
                        {"area": "Alandur", "contact": "044-22321755", "mobile": "9445850441"},
                        {"area": "Adyar", "contact": "044-24913001", "mobile": "9445850438"},
                        {"area": "Guindy", "contact": "044-22342300", "mobile": "9445850442"}
                    ]
                }
            }
        },
        
        "TNSTC": {
            "main_contact": "1800-599-1500",
            "whatsapp": "94450-14448",
            "head_office": "Thiruvalluvar House, Pallavan Salai, Chennai - 600002",
            "website": "tnstc.in",
            "services": [
                "Bus transport", "Route complaints", "Driver/conductor issues",
                "Fare disputes", "Lost items", "Service quality"
            ],
            "depots": {
                "Chennai_Division_1": ["Koyambedu", "Anna Nagar", "Ambattur", "Poonamallee"],
                "Chennai_Division_2": ["Broadway", "T.Nagar", "Adyar", "Velachery"],
                "Chennai_Division_3": ["Tambaram", "Chrompet", "Pallavaram", "Perungalathur"]
            }
        }
    },
    
    "common_issues_database": {
        "water_supply": {
            "keywords": ["water", "supply", "shortage", "leak", "pipe", "connection", "pressure", "quality", "tanker"],
            "department": "Chennai Metro Water",
            "typical_response_time": "24 hours",
            "escalation_path": "Area Engineer -> Executive Engineer -> Superintending Engineer -> Chief Engineer",
            "common_areas": ["Adyar", "Velachery", "Anna Nagar", "Thiruvottiyur", "Sholinganallur"],
            "seasonal_issues": "Summer months (March-June) - increased complaints"
        },
        
        "garbage_collection": {
            "keywords": ["garbage", "waste", "collection", "bin", "sweeping", "cleaning", "disposal"],
            "department": "Greater Chennai Corporation",
            "private_operators": {
                "Ramky_Enviro": {"zones": [1, 2, 3, 7]},
                "Urbaser_Sumeet": {"zones": [9, 10, 11, 12, 13, 14, 15]},
                "GCC_Direct": {"zones": [4, 5, 6, 8]}
            },
            "collection_schedule": "Daily door-to-door collection",
            "complaint_methods": ["1913", "Namma Chennai App", "Zone offices"]
        },
        
        "electricity": {
            "keywords": ["power", "electricity", "outage", "billing", "meter", "connection", "voltage", "transformer"],
            "department": "TANGEDCO",
            "response_times": {
                "emergency": "1 hour",
                "routine_repairs": "24 hours",
                "new_connections": "7-15 days"
            },
            "compensation": "Rs. 50 per 6 hours delay for restoration",
            "complaint_channels": ["94987-94987", "1912", "Online portal", "WhatsApp: 94458508111"]
        },
        
        "roads_infrastructure": {
            "keywords": ["road", "pothole", "street", "repair", "maintenance", "signal", "traffic"],
            "department": "Greater Chennai Corporation",
            "reporting_methods": ["1913", "Namma Chennai App", "Zone offices"],
            "categories": ["Potholes", "Street lighting", "Traffic signals", "Footpaths", "Drainage"]
        },
        
        "transport": {
            "keywords": ["bus", "transport", "route", "fare", "conductor", "driver", "schedule"],
            "department": "TNSTC",
            "complaint_types": ["Route issues", "Fare disputes", "Staff behavior", "Service quality", "Lost items"],
            "response_mechanism": "Call center -> Depot -> Route investigation"
        }
    }
}

# Create pincode mapping for all major areas
pincode_mapping = {
    "600001": "Parrys Corner, Fort",
    "600002": "Anna Road, Mount Road",
    "600003": "Park Town",
    "600004": "Mylapore",
    "600005": "Chepauk",
    "600006": "Greams Road",
    "600008": "Egmore",
    "600009": "Fort St. George",
    "600010": "Kilpauk",
    "600011": "Perambur",
    "600012": "Perambur Barracks",
    "600013": "Royapuram",
    "600014": "Royapettah",
    "600015": "Saidapet",
    "600016": "Alwarpet",
    "600017": "T. Nagar",
    "600018": "Kodambakkam",
    "600020": "Adyar",
    "600022": "Raj Bhavan",
    "600023": "Ayanavaram",
    "600024": "Kodambakkam",
    "600025": "Engineering College",
    "600026": "Nungambakkam",
    "600027": "Meenambakkam",
    "600028": "R.A. Puram",
    "600029": "Aminjikarai",
    "600030": "Shenoy Nagar",
    "600031": "Chetpet",
    "600032": "Guindy",
    "600033": "Gopalapuram",
    "600034": "Nungambakkam",
    "600035": "Nandanam",
    "600036": "IIT Madras",
    "600038": "ICF Colony",
    "600040": "Anna Nagar",
    "600043": "Pallavaram",
    "600044": "Chrompet",
    "600050": "Padi",
    "600051": "Madhavaram Milk Colony",
    "600052": "Red Hills",
    "600053": "Ambattur",
    "600054": "Avadi",
    "600055": "Avadi I.A.F.",
    "600056": "Poonamallee",
    "600057": "Ennore",
    "600058": "Ambattur Industrial Estate",
    "600060": "Madhavaram",
    "600061": "Nanganallur",
    "600062": "Sathyamurthy Nagar",
    "600063": "Perungalathur",
    "600064": "Chitlapakkam",
    "600065": "CRP Camp Avadi",
    "600066": "Puzhal",
    "600068": "Manali",
    "600069": "Kundrathur",
    "600070": "Anakaputhur",
    "600071": "Kamaraj Nagar",
    "600072": "Pattabiram",
    "600073": "Selaiyur",
    "600074": "Pozhichalur",
    "600078": "K.K. Nagar",
    "600079": "Sowcarpet",
    "600080": "Korattur",
    "600082": "Kanchipuram Road",
    "600083": "Ashok Nagar",
    "600084": "Flowers Road",
    "600085": "Kottur",
    "600086": "Gopalapuram",
    "600087": "Alwarthirunagar",
    "600088": "Adambakkam",
    "600090": "Besant Nagar",
    "600091": "Madipakkam",
    "600093": "Saligramam",
    "600094": "Choolaimedu",
    "600096": "Perungudi",
    "600097": "Ekkaduthangal",
    "600099": "Kolathur",
    "600101": "Anna Nagar West Extension",
    "600102": "Anna Nagar East",
    "600103": "Manali New Town",
    "600106": "Arumbakkam",
    "600107": "Koyambedu",
    "600108": "Broadway",
    "600116": "Porur",
    "600117": "Old Pallavaram",
    "600118": "Kodungaiyur"
}

print("✅ Complete Chennai Civic Data Structure Created")
print(f"📊 Total Zones: {chennai_complete_data['administrative_structure']['total_zones']}")
print(f"📊 Total Wards: {chennai_complete_data['administrative_structure']['total_wards']}")
print(f"📊 Departments Covered: {len(chennai_complete_data['departments_complete'])}")
print(f"📊 Pincode Mappings: {len(pincode_mapping)}")
print(f"📊 Common Issues Categories: {len(chennai_complete_data['common_issues_database'])}")

# Save to file for application use
with open('chennai_complete_civic_data.json', 'w') as f:
    json.dump(chennai_complete_data, f, indent=2)

with open('chennai_pincode_mapping.json', 'w') as f:
    json.dump(pincode_mapping, f, indent=2)

print("\n✅ Data files saved: chennai_complete_civic_data.json, chennai_pincode_mapping.json")