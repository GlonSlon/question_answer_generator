import os
import glob
import re
from typing import List, Dict, Tuple
import logging
from dataclasses import dataclass
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("nltk is not installed. I'm using a simple tokenizer.")

try:
    import spacy
    NLP = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spacy is not installed. I'm using basic rules.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn is not installed. I'm using simple heuristics.")

@dataclass
class QAPair:
    question: str
    answer: str
    source_file: str
    confidence: float = 1.0

class SimpleQAGenerator:
    
    def __init__(self):
        self.question_patterns = [
            (r'what is ([^?]+)', r'What is \1?'),
            (r'who (was|is) ([^?]+)', r'Who \1 \2?'),
            (r'when (was|did) ([^?]+)', r'When \1 \2?'),
            (r'where (is|are|was) ([^?]+)', r'Where \1 \2?'),
            (r'why (does|do|did) ([^?]+)', r'Why \1 \2?'),
            (r'how (does|do|did|many|much) ([^?]+)', r'How \1 \2?'),
        ]
        
        # Key words
        self.important_keywords = [
            'is', 'are', 'was', 'were', 'called', 'known as',
            'defined as', 'refers to', 'consists of', 'includes',
            'example', 'important', 'key', 'main', 'primary',
            'first', 'second', 'finally', 'therefore', 'thus',
            'conclusion', 'summary', 'result', 'because'
        ]
    
    def read_txt_files(self) -> Dict[str, str]:
        txt_files = glob.glob("*.txt")
        
        if "dataset_gpt.txt" in txt_files:
            txt_files.remove("dataset_gpt.txt")
        
        if not txt_files:
            raise FileNotFoundError("There are no .txt files in the current folder.")
        
        logger.info(f"Found {len(txt_files)} files: {txt_files}")
        
        documents = {}
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents[file_path] = content
                logger.info(f"{file_path} loaded: {len(content)} characters")
            except Exception as e:
                logger.error(f"Read {file_path} error: {e}")
        
        return documents
    
    def split_into_sentences(self, text: str) -> List[str]:
        if NLTK_AVAILABLE:
            return sent_tokenize(text)
        else:
            # Dot token
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if len(s.strip()) > 20]
    
    def is_definition_sentence(self, sentence: str) -> bool:
        sentence_lower = sentence.lower()
        
        # Patterns
        definition_patterns = [
            r'is a',
            r'are the',
            r'refers to',
            r'defined as',
            r'consists of',
            r'comprises',
            r'include',
            r'example of',
        ]
        
        for pattern in definition_patterns:
            if pattern in sentence_lower:
                return True
        
        return False
    
    def extract_key_terms(self, sentence: str) -> List[str]:
        if SPACY_AVAILABLE:
            doc = NLP(sentence)
            terms = [ent.text for ent in doc.ents if len(ent.text) > 3]
            terms += [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 3]
            return list(set(terms))
        else:
            words = sentence.split()
            terms = [word for word in words if word[0].isupper() or len(word) > 6]
            return terms
    
    def generate_question_from_sentence(self, sentence: str, key_term: str = None) -> str:
        if not key_term:
            terms = self.extract_key_terms(sentence)
            key_term = terms[0] if terms else None
        
        if not key_term:
            return None
        
        sentence_lower = sentence.lower()
        
        if ' is ' in sentence_lower and key_term.lower() in sentence_lower:
            return f"What is {key_term}?"
        elif ' are ' in sentence_lower and key_term.lower() in sentence_lower:
            return f"What are {key_term}?"
        elif ' was ' in sentence_lower:
            return f"What was {key_term}?"
        elif ' called ' in sentence_lower:
            return f"What is called {key_term}?"
        elif ' known as ' in sentence_lower:
            return f"What is known as {key_term}?"
        elif ' example ' in sentence_lower:
            return f"What is an example of {key_term}?"
        
        # Default
        return f"What is {key_term}?"
    
    def find_best_answer(self, sentence: str, question: str) -> str:
        return sentence
    
    def generate_qa_pairs_from_file(self, filename: str, content: str) -> List[QAPair]:
        qa_pairs = []
        sentences = self.split_into_sentences(content)
        
        logger.info(f"Processing {filename}: {len(sentences)} sentences")
        
        for i, sentence in enumerate(tqdm(sentences, desc=f"Analysis {filename}")):
            if len(sentence) < 20 or len(sentence) > 700:
                continue
            
            if self.is_definition_sentence(sentence) or any(kw in sentence.lower() for kw in self.important_keywords):
                terms = self.extract_key_terms(sentence)
                
                for term in terms[:2]:
                    question = self.generate_question_from_sentence(sentence, term)
                    
                    if question:
                        qa_pairs.append(QAPair(
                            question=question,
                            answer=sentence,
                            source_file=filename
                        ))
                        
                        # "according to" question
                        if len(qa_pairs) % 3 == 0:
                            qa_pairs.append(QAPair(
                                question=f"According to the text, {question.lower()}",
                                answer=sentence,
                                source_file=filename
                            ))
        
        return qa_pairs
    
    def filter_and_deduplicate(self, qa_pairs: List[QAPair]) -> List[QAPair]:
        unique_pairs = {}
        
        for pair in qa_pairs:
            key = pair.question.lower().strip()
            
            if len(pair.answer) < 20 or len(pair.answer) > 500:
                continue
            
            if pair.question.endswith('?'):
                if key not in unique_pairs:
                    unique_pairs[key] = pair
                else:
                    if len(pair.answer) > len(unique_pairs[key].answer):
                        unique_pairs[key] = pair
        
        return list(unique_pairs.values())
    
    def save_dataset(self, qa_pairs: List[QAPair], output_file: str = "dataset_gpt.txt"):
        
        qa_pairs.sort(key=lambda x: (x.source_file, -x.confidence))
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for pair in qa_pairs:
                f.write(f"Question: {pair.question} Answer: {pair.answer}\n")
        
        logger.info(f"Saved {len(qa_pairs)} QA pairs to {output_file}")
        
        meta_file = output_file.replace('.txt', '_meta.json')
        import json
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump([{
                'question': p.question,
                'answer': p.answer[:100] + '...',
                'source': p.source_file,
                'confidence': p.confidence
            } for p in qa_pairs], f, indent=2, ensure_ascii=False)
        
        logger.info(f"Meta information is saved in {meta_file}")
        
        stats = {}
        for pair in qa_pairs:
            stats[pair.source_file] = stats.get(pair.source_file, 0) + 1
        
        logger.info("File statistics:")
        for file, count in stats.items():
            logger.info(f"{file}: {count} pairs")

class AdvancedQAGenerator(SimpleQAGenerator):
    
    def __init__(self):
        super().__init__()
        self.use_ml = SKLEARN_AVAILABLE
        
        if self.use_ml:
            self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    
    def find_important_sentences(self, sentences: List[str]) -> List[Tuple[str, float]]:
        if not self.use_ml or len(sentences) < 2:
            return [(s, 1.0) for s in sentences]
        
        try:
            # TF-IDF
            tfidf_matrix = self.tfidf.fit_transform(sentences)
            
            # TF-IDF W
            importance_scores = tfidf_matrix.sum(axis=1).A1
            
            if importance_scores.max() > 0:
                importance_scores = importance_scores / importance_scores.max()
            
            return list(zip(sentences, importance_scores))
        except:
            return [(s, 1.0) for s in sentences]
    
    def generate_qa_pairs_from_file(self, filename: str, content: str) -> List[QAPair]:
        basic_pairs = super().generate_qa_pairs_from_file(filename, content)
        
        if not self.use_ml:
            return basic_pairs
        
        sentences = [p.answer for p in basic_pairs]
        important_sentences = self.find_important_sentences(sentences)
        
        enhanced_pairs = []
        for pair, (_, importance) in zip(basic_pairs, important_sentences):
            pair.confidence = importance
            if importance > 0.3:
                enhanced_pairs.append(pair)
        
        return enhanced_pairs

def main():
    print("\nChecking dependencies:")
    print(f"  nltk: {'+' if NLTK_AVAILABLE else '-'} (pip install nltk)")
    print(f"  spacy: {'+' if SPACY_AVAILABLE else '-'} (pip install spacy && python -m spacy download en_core_web_sm)")
    print(f"  scikit-learn: {'+' if SKLEARN_AVAILABLE else '-'} (pip install scikit-learn)")
    
    if SKLEARN_AVAILABLE:
        generator = AdvancedQAGenerator()
    else:
        generator = SimpleQAGenerator()
    
    try:
        print("\nReading files...")
        documents = generator.read_txt_files()
        
        if not documents:
            print("There are no files to process.")
            return
        
        all_qa_pairs = []
        for filename, content in documents.items():
            print(f"\nProcessing {filename}...")
            qa_pairs = generator.generate_qa_pairs_from_file(filename, content)
            all_qa_pairs.extend(qa_pairs)
            print(f"Generated {len(qa_pairs)} pairs.")
        
        print("\nFiltering and removing duplicates...")
        filtered_pairs = generator.filter_and_deduplicate(all_qa_pairs)
        print(f"After filtering: {len(filtered_pairs)} pairs.")
        
        generator.save_dataset(filtered_pairs)
        
        print("File created: dataset_gpt.txt.")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
