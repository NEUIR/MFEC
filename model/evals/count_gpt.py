import json


def calculate_sentencebert(input_path):
    with open(input_path, 'r') as f:
        score=0
        count=0
        for l in f.readlines():
            sample = json.loads(l)
            similarity_score = sample['similarity_score']
            if 're.Match' in similarity_score:
                for num in ['1', '2', '3', '4', '5']:
                    if num in similarity_score:
                        similarity_score=num
                        count = count + 1
                        score = score + int(float(similarity_score))
                        break
            elif 'To evaluate the' in similarity_score:
                continue
            else:
                count = count + 1
                score = score + int(float(similarity_score))
        score = score / count
        return score
if __name__ == '__main__':
    sentence_bert=calculate_sentencebert('./gpt/compedit_scifact.jsonl')
    print('final sentence_bert of compedit_scifact',sentence_bert)
    sentence_bert = calculate_sentencebert('./gpt/compedit_fever.jsonl')
    print('final sentence_bert of compedit_fever', sentence_bert)

    sentence_bert = calculate_sentencebert('./gpt/zerofec_scifact.jsonl')
    print('final sentence_bert of zerofec_scifact', sentence_bert)
    sentence_bert = calculate_sentencebert('./gpt/zerofec_fever.jsonl')
    print('final sentence_bert of zerofec_fever', sentence_bert)

    sentence_bert=calculate_sentencebert('./gpt/ft_scifact.jsonl')
    print('final sentence_bert of ft_scifact',sentence_bert)
    sentence_bert = calculate_sentencebert('./gpt/ft_fever.jsonl')
    print('final sentence_bert of ft_fever', sentence_bert)

    sentence_bert=calculate_sentencebert('./gpt/mfec_scifact.jsonl')
    print('final sentence_bert of mfec_scifact',sentence_bert)
    sentence_bert = calculate_sentencebert('./gpt/mfec_fever.jsonl')
    print('final sentence_bert of mfec_fever', sentence_bert)

    sentence_bert=calculate_sentencebert('./gpt/zeroshot_scifact.jsonl')
    print('final sentence_bert of zeroshot_scifact',sentence_bert)
    sentence_bert = calculate_sentencebert('./gpt/zeroshot_fever.jsonl')
    print('final sentence_bert of zeroshot_fever', sentence_bert)

    sentence_bert=calculate_sentencebert('./gpt/fec_scifact.jsonl')
    print('final sentence_bert of fec_scifact',sentence_bert)
    sentence_bert = calculate_sentencebert('./gpt/fec_fever.jsonl')
    print('final sentence_bert of fec_fever', sentence_bert)
