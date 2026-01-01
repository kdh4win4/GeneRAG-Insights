import pandas as pd
import re

class VariantProcessor:
    """
    A class to parse and preprocess genetic variant data (VCF/CSV).
    """
    def __init__(self):
        self.supported_formats = ['.vcf', '.csv', '.txt']

    def parse_variant_string(self, variant_text):
        """
        Extracts gene name and mutation details from a raw string.
        Example input: "EGFR T790M" -> Output: {"gene": "EGFR", "variant": "T790M"}
        """
        # Basic regex to separate Gene name and Mutation pattern
        match = re.match(r"([A-Z0-9]+)\s+(.+)", variant_text.strip())
        if match:
            return {
                "gene": match.group(1),
                "variant": match.group(2)
            }
        return {"error": "Invalid format. Please use 'Gene Mutation' format (e.g., BRAF V600E)."}

    def format_query_for_rag(self, parsed_data):
        """
        Converts parsed variant data into a professional search query for RAG.
        """
        if "error" in parsed_data:
            return parsed_data["error"]
        
        query = (f"What is the clinical significance of the {parsed_data['gene']} "
                 f"{parsed_data['variant']} mutation in precision oncology?")
        return query

# Example Usage (Commented out for production)
# if __name__ == "__main__":
#     proc = VariantProcessor()
#     print(proc.parse_variant_string("BRCA1 C61G")) 
