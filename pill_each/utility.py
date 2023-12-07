import ahocorasick


class WordExtract:
    def __init__(self, keywords):
        self.keywords = keywords  # type(list)  list id equals to keyword id
        self.automat = self.build_automat()

    def build_automat(self):
        auto = ahocorasick.Automaton()
        for index, keyword in enumerate(self.keywords):
            auto.add_word(keyword, (index, keyword))
        auto.make_automaton()
        return auto

    def extract_keywords(self, text):
        keyword_matches = set()
        for end_index, (keyword_index, original_keyword) in self.automat.iter(text):
            start_index = end_index - len(original_keyword) + 1
            keyword_matches.add((keyword_index, start_index, end_index))
        return keyword_matches


if __name__ == "__main__":
    # 示例文本和关键词列表
    sample_text = "这是一个示例文本，其中包含一些关键词，如Python、关键字、文本处理等如Python。"
    sample_keywords = ["关键词", "Python", "文本处理"]

    WE = WordExtract(sample_keywords)

    # 提取关键词
    result = WE.extract_keywords(sample_text)

    # 打印结果
    print("文本中的关键词:", result)
