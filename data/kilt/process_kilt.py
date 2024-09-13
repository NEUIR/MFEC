import json

from tqdm import tqdm
wiki_dict = dict()
with open('../data/kilt/kilt_knowledgesource.json', 'r') as f:
    for line in  tqdm(f):
        data = json.loads(line)
        id = data["wikipedia_id"]
        text = data['text']
        wiki_dict[id] = {"wikipedia_id": id, "text": text}
input_path_list = ['./nq-dev-kilt.jsonl','./nq-train-kilt.jsonl']
output_path_list = ['./nq-dev-kilt_outputs.jsonl','./nq-train-kilt_outputs.jsonl']
for i in range(len(input_path_list)):
    with open(input_path_list[i], 'r') as f:
        output_list = []
        for l in tqdm(f):
            data = json.loads(l)
            generated_question = []
            answer = []
            step = 0
            for item in data['output']:
                section_list = []
                if 'provenance' in item and 'answer' not in item:
                    provenance_list = item['provenance']
                    for provenance in provenance_list:
                        section = provenance['section']
                        wikipedia_id = provenance['wikipedia_id']
                        start_paragraph_id = provenance['start_paragraph_id']
                        end_paragraph_id = provenance['end_paragraph_id']
                        if 'start_character' in provenance and 'end_character' in provenance:
                            start_character = provenance['start_character']
                            end_character = provenance['end_character']
                        else:
                            start_character = 0
                            end_character = -1
                        instance = {'wikipedia_id': wikipedia_id, 'section': section,
                                    'start_paragraph_id': start_paragraph_id, 'start_character': start_character,
                                    'end_paragraph_id': end_paragraph_id, 'end_character': end_character}
                        section_list.append(instance)
            for item in data['output']:
                if 'answer' in item and 'provenance' in item:
                    if item['answer'] not in answer:
                        answer.append(item['answer'])
                        generated_question = data['input']
                        provenance_list = item['provenance']
                        evidence_list = []
                        for provenance in provenance_list:
                            section = provenance['section']
                            wikipedia_id = provenance['wikipedia_id']
                            wiki = wiki_dict[wikipedia_id]
                            text = wiki['text']
                            for evi in section_list:
                                if evi['wikipedia_id'] == wikipedia_id and evi['section'] == section:
                                    start_paragraph_id = evi['start_paragraph_id']
                                    start_character = evi['start_character']
                                    end_paragraph_id = evi['end_paragraph_id']
                                    end_character = evi['end_character']
                                    break
                            evidence = ''
                            for idx, paragraph in enumerate(text):
                                if start_paragraph_id == end_paragraph_id:
                                    if idx == start_paragraph_id:
                                        if start_character == 0 and end_character == -1:
                                            evidence = paragraph[start_character:]
                                        else:
                                            evidence = paragraph[start_character:end_character]
                                        break
                                if start_paragraph_id != end_paragraph_id:
                                    if idx == start_paragraph_id:
                                        evidence = evidence + paragraph[start_character:]
                                    if idx > start_paragraph_id and idx < end_paragraph_id:
                                        evidence = evidence + paragraph
                                    if idx == end_paragraph_id:
                                        evidence = evidence + paragraph[:end_character]
                                        break
                            evidence_list.append(evidence)

                        instance = {'answer': item['answer'], 'generated_question': data['input'],
                                        'evidence': evidence_list}
                        if len(evidence_list)!=0:
                            output_list.append(instance)
            step = step+1
    with open(output_path_list[i], 'w') as fout:
        for data in output_list:
            fout.write(json.dumps(data)+'\n')
