import nltk
from nltk.tokenize import word_tokenize
import string
import re
import pandas as pd
from nltk.corpus import cmudict
import pandas as pd
import re

# download third-party dependencies
nltk.download('cmudict')
nltk.download('punkt')
nltk.download('punkt_tab')


class METERChecker(object):
    @staticmethod
    def is_iambic_pentameter(meter_pattern):
        if len(meter_pattern) != 10:
            return False

        expected_pattern = ['u', 'S'] * 5  # 5 iambs
        matches = sum(1 for i in range(10) if meter_pattern[i] == expected_pattern[i] or meter_pattern[i] == '*')

        return matches / 10 >= 0.7  # at least 70% match

    @staticmethod
    def is_iambic_tretameter(meter_pattern):
        if len(meter_pattern) != 8:
            return False

        expected_pattern = ['u', 'S'] * 4  # 4 iambs
        matches = sum(1 for i in range(8) if meter_pattern[i] == expected_pattern[i] or meter_pattern[i] == '*')

        return matches / 8 >= 0.7  # at least 70% match

    @staticmethod
    def is_iambic_trimeter(meter_pattern):
        if len(meter_pattern) != 6:
            return False

        expected_pattern = ['u', 'S'] * 3  # 3 iambs
        matches = sum(1 for i in range(6) if meter_pattern[i] == expected_pattern[i] or meter_pattern[i] == '*')

        return matches / 6 >= 0.7  # at least 70% match


class POEMFormat(object):
    def __init__(self, ):
        self.valid_formats = ["limerick", "sonnet", "ballad", "ghazal", "pantoum", "villanelle", "sestina"]
        self.meter_checker = METERChecker()

    def is_limerick(self, meter_analysis, rhyme_analysis):
        lines = len(rhyme_analysis)

        if lines == 5:
            # check AABBA rhyme pattern
            if (rhyme_analysis[0][2] == rhyme_analysis[1][2] == rhyme_analysis[4][2]):
                return True
        elif lines == 4:
            # check AABA rhyme pattern
            if (rhyme_analysis[0][2] == rhyme_analysis[1][2] == rhyme_analysis[3][2]):
                third_line_parts = rhyme_analysis[2][0].split(',')
                if len(third_line_parts) == 2:
                    rhyme1 = extract_rhyme_foot(third_line_parts[0].strip())
                    rhyme2 = extract_rhyme_foot(third_line_parts[1].strip())
                    if rhymes_similar(rhyme1, rhyme2):
                        return True
        return False

    def is_sonnet(self, meter_analysis, rhyme_analysis):
        # we don't need to pre-check sonnets
        return True

    def is_ballad(self, meter_analysis, rhyme_analysis):
        sections = []
        section = []

        for line, _, rhyme_letter in rhyme_analysis:
            section.append((line, rhyme_letter))
            if len(section) == 4:
                sections.append(section)
                section = []

        if section:
            sections.append(section)

        total_sections = len(sections)
        conforming_sections = 0

        # check ABCB/ABAB rhyme patterns
        for sec in sections:
            if len(sec) != 4 or sec[1][1] != sec[3][1]:
                return False

            conforming_sections += 1

        if conforming_sections / total_sections >= 0.7: # at least 70% match
            return True

        return False

    def is_ghazal(self, meter_analysis, rhyme_analysis):
        # preprocessing
        processed = []
        for line, _, _ in rhyme_analysis:
            # delete end-of-sentence punc and tokenize
            cleaned_line = line.rstrip(string.punctuation)
            words = [w.rstrip(string.punctuation) for w in cleaned_line.split()]
            processed.append(words)

        # pre-check
        if len(processed) < 2 or len(processed) % 2 != 0:
            return False

        # check the rhyme pattern in the first couplet
        try:
            # check first line
            line1 = processed[0]
            radif_word = line1[-1].lower()  # get Radif
            qafiya_word = line1[-2]  # get Qafiya

            # check second line
            line2 = processed[1]
            if line2[-1].lower() != radif_word:
                return False
            if not rhymes_similar(line2[-2], qafiya_word):
                return False
        except IndexError:
            return False

        # check other couplets
        for i in range(2, len(processed), 2):

            if i + 1 >= len(processed):
                break

            current_line = processed[i + 1]  # only check the second line of every couplet
            try:
                # check Radif
                if current_line[-1].lower() != radif_word:
                    return False

                # check Qafiya
                if not rhymes_similar(current_line[-2], qafiya_word):
                    return False
            except IndexError:
                return False

        return True   

    def is_pantoum(self, meter_analysis, rhyme_analysis):
        # pre-check
        if (len(rhyme_analysis) < 8) or (len(rhyme_analysis) % 4 != 0):
            return False

        # divide quatrains (4-line stanzas)
        stanzas = []
        current_stanza = []

        for index, item in enumerate(meter_analysis):
            words = item[0]
            current_stanza.append(words)

            if (index + 1) % 4 == 0:
                stanzas.append(current_stanza)  # add to stanzas
                current_stanza = []

        # check repetition
        for i in range(len(stanzas) - 1):
            if (stanzas[i][1] != stanzas[i + 1][0]) or \
                    (stanzas[i][3] != stanzas[i + 1][2]):
                return False

        # check first and last quatrain repetiion
        return (stanzas[-1][1] == stanzas[0][0]) and \
            (stanzas[-1][3] == stanzas[0][2])

    def is_villanelle(self, meter_analysis, rhyme_analysis):

        if meter_analysis is None or rhyme_analysis is None:
            return False

        # pre-check
        if len(rhyme_analysis) != 19:
            return False

        # check the first 5 tercets (3-line stanzas)
        for i in range(5):
            tercet = rhyme_analysis[i * 3:(i + 1) * 3]

            if not (tercet[0][2] == tercet[2][2]):  # check rhyme
                return False

        # check the last quatrain
        quatrain = rhyme_analysis[15:19]

        if not (quatrain[0][2] == quatrain[2][2] and quatrain[0][2] == quatrain[3][2] and quatrain[2][2] == quatrain[3][2]):
            return False

        # check repetition
        first_line = rhyme_analysis[0][0]
        third_line = rhyme_analysis[2][0]

        for j in range(5):
            if rhyme_analysis[j * 3 + 2][0] != first_line and rhyme_analysis[j * 3 + 2][0] != third_line:
                return False

        differences_1 = sum(1 for a, b in zip(rhyme_analysis[17][0], first_line) if a != b)
        differences_2 = sum(1 for a, b in zip(rhyme_analysis[17][0], third_line) if a != b)
        if differences_1 > 1 and differences_2 > 1:
            return False

        return True

    def is_sestina(self, meter_analysis, rhyme_analysis):
        if meter_analysis is None or rhyme_analysis is None:
            return False

        # pre-check
        if len(rhyme_analysis) != 39:
            return False

        # check the first 6 6-line stanzas
        end_words = []
        for i in range(6):
            stanza = rhyme_analysis[i * 6:(i + 1) * 6]

            # get the ending word
            end_word = []
            for line in stanza:
                end_word.append(line[0][-1])

            # check if each stanza contains 6 lines
            if len(stanza) != 6:
                return False

            end_words.append(end_word)

        # check the order of the ending words in the last 5 stanzas
        for i in range(1, 6):
            current_end_words = end_words[i]

            # check if re-ordered
            if sorted(current_end_words) != sorted(end_words[0]):
                return False

        return True

    def check_poem(self, meter_analysis, rhyme_analysis, meter, rhyme, form):
        def transform_string(s):
            char_map = {}
            result = []
            next_char = 'A'

            for char in s:
                if char not in char_map:
                    char_map[char] = next_char
                    next_char = chr(ord(next_char) + 1)  # Move to the next character
                result.append(char_map[char])

            return ''.join(result)

        # precheck via rule-based methods
        assert form in self.valid_formats, f"invalid format {form} not in {self.valid_formats}."
        form_check_func = getattr(self, f"is_{form}")
        form_check_ret = form_check_func(meter_analysis, rhyme_analysis)
        if isinstance(form_check_ret, tuple):
            if not form_check_ret[0]:
                return False, "Fail in pre-check"

        if meter:
            func_name = "is_" + meter.lower().replace(" ", "_")
            meter_check_func = getattr(self.meter_checker, func_name)

            # record the number of lines that didn't satisfy the meter pattern
            non_meter_count = 0
            poem_meters = []
            for _, _, poem_meter in meter_analysis:
                poem_meters.append(poem_meter)
                if not meter_check_func(poem_meter):
                    non_meter_count += 1

            if non_meter_count / len(meter_analysis) >= 0.3:
                return False, f"False meter pattern:\npoem_meter: {poem_meters}\nmeter: {meter}"

        if rhyme:
            poem_rhyme = "".join(r[-1] for r in rhyme_analysis)
            if len(poem_rhyme) % len(rhyme) != 0:
                return False, f"False number of lines. \npoem_rhyme: {poem_rhyme}\nrhyme: {rhyme}"
            n_line_per_group = len(rhyme)
            n_group = len(poem_rhyme) // len(rhyme)
            n_matched = 0
            new_poem_rhymes = ""
            for group_index in range(n_group):
                poem_rhyme_cur_group = poem_rhyme[group_index * n_line_per_group:(group_index + 1) * n_line_per_group]
                new_poem_rhymes += transform_string(poem_rhyme_cur_group)
                for pr, r in zip(transform_string(poem_rhyme_cur_group), rhyme):
                    if pr == r:
                        n_matched += 1

            if n_matched / len(poem_rhyme) > 0.7:
                return True, ""
            return False, f"False rhyme pattern. \npoem_rhyme: {new_poem_rhymes}\nrhyme: {rhyme}"
        return True, ""


class POEMetrics(object):
    """
    Data format:
    id, form, meter, rhyme, model1_poem, model2_poem, ...
    """
    def __init__(self, ):
        # Parse file into standard format (list of dict)
        self.cmudict = cmudict.dict()
        self.poem_format = POEMFormat()

    @staticmethod
    def poem2words(poem: str) -> list[list[str]]:
        # Split the poem into lines
        lines = poem.strip().split('\n')
        lines = [line.strip() for line in lines if line.strip()]

        # Split each line into words and create a list of lists
        def line2words(l: str) -> list[str]:
            # Replace hyphens and possessive forms
            l = l.replace("-", " ").replace("'s ", " ")
            # Tokenize the line
            tokens = word_tokenize(l)
            # Remove the token "s" if it exists
            tokens = [token for token in tokens if token != "s" and token.isalpha()]
            return tokens

        word_lists = [line2words(line) for line in lines]
        word_lists = [words for words in word_lists if len(words) > 0]
        # for line_words in word_lists:
        #     assert len(line_words) > 0, f"Empty line in {poem}"
        return word_lists

    def poem2rhyme(self, poem_words: list[list[str]]) -> list[tuple]:
        
        def rhymes_similar(foot1, foot2):
            """check rhyme"""
            if foot1 == foot2:
                return True

            # ignore stress
            foot1_rhyme = re.sub(r'\d', '', foot1)
            foot2_rhyme = re.sub(r'\d', '', foot2)

            len1, len2 = len(foot1_rhyme), len(foot2_rhyme)

            # if of the same length, allow 1-letter diff
            if len1 == len2:
                diff_count = sum(1 for a, b in zip(foot1_rhyme, foot2_rhyme) if a != b)
                if diff_count == 1:
                    return True

            # if length diff =  +/- 1, allow 1-letter diff
            if abs(len1 - len2) == 1:
                longer, shorter = (foot1_rhyme, foot2_rhyme) if len1 > len2 else (foot2_rhyme, foot1_rhyme)
                for i in range(len(shorter) + 1):
                    if longer[:i] + longer[i + 1:] == shorter:
                        return True

            if foot1_rhyme == foot2_rhyme:
                return True

            return False

        def extract_rhyme_foot(word):
            if word.lower() in self.cmudict:
                phones = self.cmudict[word.lower()][0]
                vowels = re.compile(r'[AEIOU]')
                rhyme_parts = []

                for phone in reversed(phones):
                    if vowels.search(phone):
                        rhyme_parts.append(phone)
                        break
                    rhyme_parts.append(phone)

                return ''.join(reversed(rhyme_parts))
            return ''

        last_word_lists = [line_words[-1] for line_words in poem_words]
        last_word_rhyme_feet = [extract_rhyme_foot(word) for word in last_word_lists]
        rhyme_mapping = {}
        rhyme_letters = []
        rhyme_counter = 0
        for last_word_rhyme_foot in last_word_rhyme_feet:
            found_rhyme_letter = None
            for existing_foot, rhyme_letter in rhyme_mapping.items():
                if rhymes_similar(existing_foot, last_word_rhyme_foot):
                    found_rhyme_letter = rhyme_letter
                    break

            if found_rhyme_letter:
                rhyme_letter = found_rhyme_letter
            else:
                rhyme_counter += 1
                rhyme_letter = chr(64 + rhyme_counter)  # A, B, C, B, ...
                rhyme_mapping[last_word_rhyme_foot] = rhyme_letter
            rhyme_letters.append(rhyme_letter)

        return [(w, rf, rl) for w, rf, rl in zip(last_word_lists, last_word_rhyme_feet, rhyme_letters)]

    def poem2meter(self, poem_words: list[list[str]]) -> list[tuple]:
        
        def get_syllables(word):
            if word.lower() in self.cmudict:
                return [len(list(y for y in x if y[-1].isdigit())) for x in self.cmudict[word.lower()]]
            return [0]

        def is_monophthongal(phones):
            """check if the phone is monophthongal"""
            vowel_count = sum(1 for phone in phones if re.search(r'[AEIOU]', phone))
            return vowel_count == 1

        def get_meter_pattern(phones):
            pattern = []
            for stress in phones:
                if stress[-1].isdigit():
                    if stress[-1] in ['0', '2']:
                        pattern.append('u')  # unstressed
                    elif stress[-1] == '1':
                        pattern.append('S')  # stressed

            # monophthong can be either stressed or unstressed, marked as '*'
            if is_monophthongal(phones):
                if pattern:
                    pattern[-1] = '*'
            return pattern

        meter_analysis = []
        for words_per_line in poem_words:
            syllable_count = 0
            meter_pattern = []
            for word in words_per_line:
                syllables = get_syllables(word)
                if syllables[0] > 0: 
                    syllable_count += syllables[0]
                    phones = self.cmudict[word.lower()][0]
                    meter_pattern.extend(get_meter_pattern(phones))
                else:
                    print(f"Warning: No pronunciation found for '{word}'")
                    pass
            meter_analysis.append(
                (words_per_line, syllable_count, meter_pattern))
        return meter_analysis

    def evaluate_row(self, row):
        reserved_keys = ["id", "Form", "Meter", "Rhyme", "prompt"]
        if "prompt" in row:
            form, meter, rhyme = self.extract_poem_requirements(row["prompt"])
        else:
            form = row["Form"].lower()
            meter = row["Meter"]
            rhyme = row["Rhyme"]
        form = form.lower()
        if rhyme:
            rhyme = "".join(letter for letter in rhyme if letter != " ")
        poems = {}
        for k, v in row.items():
            if k not in reserved_keys and "poem" in k.lower():
                poems[k] = v
        for poem_k, poem in poems.items():
            poem_words = self.poem2words(poem)
            poem_rhyme = self.poem2rhyme(poem_words)
            poem_meter = self.poem2meter(poem_words)
            is_correct, reason = self.poem_format.check_poem(poem_meter, poem_rhyme, meter, rhyme, form)
            row[f"{poem_k}_is_correct"] = is_correct
            row[f"{poem_k}_reason"] = reason
        return row

    def extract_poem_requirements(self, prompt):
        form_match = re.search(r'Form:\s*(.*?)\s*\n', prompt)
        meter_match = re.search(r'Meter:\s*(.*?)\s*\n', prompt)
        rhyme_match = re.search(r'Rhyme:\s*(.*?)\s*\n', prompt)

        form = form_match.group(1) if form_match else None
        meter = meter_match.group(1) if meter_match else None
        rhyme = rhyme_match.group(1) if rhyme_match else None

        return form, meter, rhyme

    def evaluate_file(self, file_path):
        # Load XLSX and convert to list of dict
        self.data = pd.read_excel(file_path).to_dict(orient='records')
        new_data = []
        for row in self.data:
            new_row = self.evaluate_row(row)
            new_data.append(new_row)
        
        # calculate the ratio of True in each row whose name contains "is_correct"
        column_names = self.data[0].keys()
        for col in column_names:
            if 'is_correct' in col:
                true_count = sum(row[col] is True for row in new_data)
                total_count = len(new_data)
                ratio = true_count / total_count if total_count > 0 else 0
                print(col, ": ", ratio)

        # Save new data to another file
        sorted_data = []
        for row in new_data:
            poem_keys = sorted(k for k in row if 'poem' in k)
            other_keys = sorted(k for k in row if 'poem' not in k)

            sorted_row = {k: row[k] for k in poem_keys + other_keys}
            sorted_data.append(sorted_row)

        output_file_path = file_path.replace(".xlsx", "_result.xlsx")
        df = pd.DataFrame(sorted_data, index=range(len(sorted_data)))
        df.to_excel(output_file_path, index=False)
        print(f"Saved to {output_file_path}")

if __name__ == '__main__':
    poemetrics = POEMetrics()
    file_path = "./all_llm_poem_sets.xlsx"
    poemetrics.evaluate_file(file_path)
