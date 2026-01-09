from collections import Counter

class AnswerGenerator:
    """
    Generates rule-based answers based on retrieved chunks.
    """
    def __init__(self):
        pass

    def generate_answer(self, query, retrieved_chunks):
        """
        Generates an answer based on the query and retrieved chunks.
        """
        if not retrieved_chunks:
            return "I couldn't find any relevant complaints matching your query."

        # Analyze metadata
        products = [chunk.get('product', 'Unknown') for chunk in retrieved_chunks]
        issues = [chunk.get('issue', 'Unknown') for chunk in retrieved_chunks]
        
        top_product = Counter(products).most_common(1)[0][0]
        top_issue = Counter(issues).most_common(1)[0][0]
        
        # Construct answer
        answer = f"Based on the analysis of customer complaints, the most relevant information pertains to **{top_product}** with issues related to **{top_issue}**.\n\n"
        
        answer += "Here are some specific details from similar complaints:\n"
        
        for i, chunk in enumerate(retrieved_chunks[:3]): # Summary of top 3
            doc_preview = chunk.get('document', '')[:200] + "..." if len(chunk.get('document', '')) > 200 else chunk.get('document', '')
            answer += f"- Complaint about {chunk.get('product')}: \"{doc_preview}\"\n"
            
        answer += "\nThis information is based on historical complaint data."
        
        return answer
