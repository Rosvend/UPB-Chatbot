"""
Metadata Manager for UPB Programs
Provides fast access to program information without retrieval
"""

import json
from pathlib import Path
from typing import List, Dict, Optional


class MetadataManager:
    """Manages program metadata for quick lookups and comprehensive queries"""
    
    def __init__(self, metadata_path: Optional[str] = None):
        if metadata_path is None:
            metadata_path = Path(__file__).parent / "programs_index.json"
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.programs = self.data['programs']
        self.metadata = self.data['metadata']
    
    def get_all_programs(self) -> List[Dict]:
        """Get list of all programs"""
        return self.programs
    
    def get_program_names(self) -> List[str]:
        """Get just the program names"""
        return [p['name'] for p in self.programs]
    
    def get_total_count(self) -> int:
        """Get total number of programs"""
        return self.metadata['total_programs']
    
    def get_abet_programs(self) -> List[str]:
        """Get programs with ABET accreditation"""
        return self.metadata.get('abet_accredited', [])
    
    def get_alta_calidad_programs(self) -> List[str]:
        """Get programs with Alta Calidad accreditation"""
        return self.metadata.get('alta_calidad_accredited', [])
    
    def search_by_keyword(self, keyword: str) -> List[Dict]:
        """Find programs matching a keyword"""
        keyword_lower = keyword.lower()
        matches = []
        
        for program in self.programs:
            # Check in name
            if keyword_lower in program['name'].lower():
                matches.append(program)
                continue
            
            # Check in keywords
            for kw in program.get('keywords', []):
                if keyword_lower in kw.lower():
                    matches.append(program)
                    break
        
        return matches
    
    def get_program_by_file(self, filename: str) -> Optional[Dict]:
        """Get program info by filename"""
        for program in self.programs:
            if program['file'] == filename:
                return program
        return None
    
    def is_accreditation_query(self, query: str) -> bool:
        """Detect if query is about accreditations"""
        accreditation_keywords = [
            'acreditación', 'acreditacion', 'abet', 'alta calidad',
            'acreditad', 'certificación', 'certificacion'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in accreditation_keywords)
    
    def is_comprehensive_query(self, query: str) -> bool:
        """Detect if query is asking for all programs"""
        query_lower = query.lower()
        
        # First check if it's an accreditation query (higher priority)
        accreditation_keywords = [
            'acreditación', 'acreditacion', 'abet', 'alta calidad',
            'acreditad', 'certificación', 'certificacion'
        ]
        if any(keyword in query_lower for keyword in accreditation_keywords):
            return False
            
        comprehensive_keywords = [
            'todas', 'todos', 'cuántas', 'cuantas', 'lista', 
            'listar', 'enumera', 'qué ingenierías', 'que ingenierias',
            'qué programas', 'que programas', 'catálogo', 'catalogo'
        ]
        
        return any(keyword in query_lower for keyword in comprehensive_keywords)
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in accreditation_keywords)
    
    def format_programs_list(self) -> str:
        """Format all programs as a numbered list"""
        programs_list = "\n".join([
            f"{i+1}. {program['name']}" 
            for i, program in enumerate(self.programs)
        ])
        return f"La UPB ofrece {self.get_total_count()} programas de ingeniería:\n\n{programs_list}"
    
    def format_abet_programs(self) -> str:
        """Format ABET accredited programs"""
        programs = self.get_abet_programs()
        if not programs:
            return "No hay información sobre programas con acreditación ABET."
        
        programs_list = "\n".join([f"- {p}" for p in programs])
        return f"Programas con acreditación ABET:\n{programs_list}"
    
    def format_alta_calidad_programs(self) -> str:
        """Format Alta Calidad accredited programs"""
        programs = self.get_alta_calidad_programs()
        if not programs:
            return "No hay información sobre programas con acreditación de Alta Calidad."
        
        programs_list = "\n".join([f"- {p}" for p in programs])
        return f"Programas con acreditación de Alta Calidad:\n{programs_list}"


def test_metadata_manager():
    """Test the metadata manager"""
    print("=" * 80)
    print("Testing MetadataManager")
    print("=" * 80)
    
    manager = MetadataManager()
    
    print(f"\n✓ Total programs: {manager.get_total_count()}")
    
    print(f"\n✓ All program names:")
    for name in manager.get_program_names():
        print(f"  - {name}")
    
    print(f"\n✓ ABET accredited programs:")
    for name in manager.get_abet_programs():
        print(f"  - {name}")
    
    print(f"\n✓ Search 'sistemas':")
    matches = manager.search_by_keyword('sistemas')
    for match in matches:
        print(f"  - {match['name']}")
    
    print(f"\n✓ Is comprehensive query?")
    test_queries = [
        "¿Cuántas ingenierías hay?",
        "Lista todas las ingenierías",
        "¿Qué es ingeniería de sistemas?"
    ]
    for q in test_queries:
        result = manager.is_comprehensive_query(q)
        print(f"  '{q}' -> {result}")
    
    print(f"\n✓ Is accreditation query?")
    test_queries = [
        "¿Qué programas tienen ABET?",
        "Acreditación de alta calidad",
        "¿Cuánto dura la carrera?"
    ]
    for q in test_queries:
        result = manager.is_accreditation_query(q)
        print(f"  '{q}' -> {result}")
    
    print(f"\n✓ Formatted programs list:")
    print(manager.format_programs_list())
    
    print(f"\n✓ Formatted ABET programs:")
    print(manager.format_abet_programs())


if __name__ == "__main__":
    test_metadata_manager()