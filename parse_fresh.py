import requests, json, os, time, tempfile, re
from datetime import date
from typing import Any, Callable, Tuple, List
from dataclasses import dataclass, asdict, is_dataclass
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from collections.abc import Generator
from functools import wraps



load_dotenv()

OPENROUTER              = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_HEADERS      = {
    "Authorization": f"Bearer {os.getenv('API_OPENROUTER')}",
    "X-Title": "agent-course-homework01/1.0 (educational project)", 
}
WEAK_MODEL              = "openai/gpt-4o-mini"
STRONG_MODEL            = "anthropic/claude-sonnet-4.6"
WIKI                    = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS            = {"User-Agent": "agent-course-homework01/1.0 (educational project)"}
SEARCH_ACTUAL_ARTICLES  = f"creationdate:{date.today().year}-{date.today().month}"
INSTRUCT_GENERATE_QUERY = f"""
Based on the information provided, you need to formulate a question, the answer to which is contained in this text. 
Requirements for each question: the event occurred after January 1, 2025 and up
to today (today is {str(date.today())}), the question is self-contained (makes it clear which event is
in question, without reading the source), the answer is short and unambiguous (name, number,
title, date - not "tell me what", not "why"), the answer is literally contained in a quote from
the source, and the quote is in the source itself. It is better to take information from the middle of the incoming text.
If there is no way to formulate a question according to the criteria, then mark the response field as 'successful' lie, 
and in the 'evidence' field, write why the request could not be generated, otherwise it is true.
"""
SOURCE                 = "fresh"
DATASET_FILE_NAME      = 'my_fresh_data.jsonl'
LOG_FILE               = 'log.jsonl'


CLIENT_OPENAI = OpenAI(
    api_key            = os.getenv('API_TOOKEN_CLUB'),
    base_url           = "https://tooken.club/v1",
)

class AnswerToQuestion(BaseModel):
    reasoning   : str  = Field(description="brief reasoning")
    final       : str  = Field(description="short final answer only, for example: 'Miguel Díaz-Canel' or '22 March'; if you do not know, write 'unknown'")

SCHEMA_PROMPT       = f"""
Answer strictly with one JSON object matching this schema, no text 
around, no LaTeX:\n{json.dumps(AnswerToQuestion.model_json_schema())}
"""


class GeneratedQuestion(BaseModel):
    question    : str  = Field(description="self-generated question")
    answer      : str  = Field(description="the answer to the generated question")
    evidence    : str  = Field(description="a quote from the text that contains the answer to the question")
    success     : bool = Field(description="was it possible to form a question")

@dataclass
class LogModelDomain:
    model           : str
    question        : str
    answer          : str
    cost            : float
    right           : bool
    reasoning       : str 

@dataclass
class QuestionDomain:
    id              : str
    source          : str
    page            : str
    url             : str
    question        : str
    answer          : str
    evidence        : str
    sonnet_answer   : str = ""
    agent_ok        : str = ""


def questionFromLLM(
        text:       str, 
        instruct:   str | None = INSTRUCT_GENERATE_QUERY
) -> GeneratedQuestion:
    response = CLIENT_OPENAI.responses.parse(
        model           = "gpt-6-luna",
        instructions    = instruct,
        input           = text,
        reasoning       = {"effort": "medium"},
        text_format     = GeneratedQuestion,
        store           = False,
    )

    return response.output_parsed


def wiki(
    key_phrase  : str          = SEARCH_ACTUAL_ARTICLES, 
    title       : str | None   = None,
    exintro     : int | None   = None,
    gsroffset   : int          = 0,
) -> Any:
    params = {
        "action"        : "query",
        "prop"          : "info|extracts",
        "explaintext"   : 1,
        "exintro"       : exintro,
        "inprop"        : "url",
        "format"        : "json",
    }

    if key_phrase and title:
        raise ValueError("Укажите только один параметр: key_phrase или title")

    if title: #если ищем статьи по заголовку для проверки моделей на датасете
        params["title"] = title
    else:     #при создании вопросов
        params.update({
            "generator"     : "search",
            "gsrlimit"      : 1,
            "gsrsearch"     : key_phrase,
            "gsroffset"     : gsroffset,
        })

    for attempt in range(3):
        try:
            r = requests.get(WIKI, params=params, headers=WIKI_HEADERS, timeout=15)

            if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))

    raise RuntimeError("Википедия недоступна")

def jsonFrom(text):
    m = re.search(r"\[.*\]|\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("нет JSON")
    cleaned = re.sub(r'(\\["\\/bfnrtu])|\\', lambda e: e.group(1) or "\\\\", m.group(0))
    return json.loads(cleaned, strict=False)


def findingAnswer(
    question    :str,
    model       :str = STRONG_MODEL, 
    temperature :int = 0
) -> Tuple[str, float]:
    body = {
        "model":model,
        "messages": [
            {"role": "system", "content": SCHEMA_PROMPT},
            {"role": "user", "content": question}
        ],
        "temperature": temperature, 
        "usage": {"include": True},
        "max_completion_tokens": 3000,
    }

    problem = "нет ответа"
    for attempt in range(3):
        try:
            r = requests.post(
                OPENROUTER, 
                json=body, 
                headers=OPENROUTER_HEADERS,
                timeout=120
            )
            if r.status_code == 200:
                data = r.json()
                return data['choices'][0]['message']['content'], data['usage']['cost']
        except requests.RequestException as e:
            problem = type(e).__name__
        time.sleep(min(60, 3 * 2 ** attempt))
    raise RuntimeError(problem)


def updateOutFile(row:str, filename: str = DATASET_FILE_NAME):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with tempfile.NamedTemporaryFile(
                    mode        = 'w',
                    encoding    = 'utf-8',
                    dir         = os.path.dirname(os.path.abspath(filename)),
                    delete      = False,
                    suffix      = '.jsonl'
                ) as temp_file:
                    temp_filename = temp_file.name
            
                    with open(filename, 'r', encoding='utf-8') as source_file:
                        for line_num, line in enumerate(source_file, start=1):
                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError as e:
                                raise ValueError(f"Некорректный JSON в строке {line_num}") from e

                            ans = func(
                                record['question'], 
                                record['answer'],
                                *args, 
                                **kwargs
                            )

                            record[row] = ans

                            json.dump(record, temp_file, ensure_ascii=False)
                            temp_file.write('\n')

                            print(f"{line_num} - готов!")                              
            
                os.replace(temp_filename, filename)
                temp_filename = None
            finally:
                if temp_filename and os.path.exists(temp_filename):
                    os.unlink(temp_filename)
        return wrapper
    return decorator


@updateOutFile('sonnet_answer')
def modelSolution(
    question    : str, 
    answer      : str, 
    model       : str,
    logList     : List[LogModelDomain] | None = None,
) -> str:
    outModel, cost = findingAnswer(question, model)

    try:
        outModelObj = AnswerToQuestion.model_validate(
            jsonFrom(outModel)
        )
        ans = outModelObj.final
        reasoning = outModelObj.reasoning

        if logList is not None:
            if ans != 'unknown':
                right = answer == ans
            else:
                right = False

            rec = LogModelDomain(
                model,
                question,
                ans,
                cost,
                right,
                reasoning,
            )
            logList.append(rec)

        return ans
    except (ValidationError, ValueError):
        pass
    
    #наша строка (которую вряд ли напишет модель), чтоб понимать, что произошла какая-то проблема
    return "problemXiX" 


def writeOutFile(
    filename: str = DATASET_FILE_NAME
):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            generator = func(*args, **kwargs)
            
            with open(filename, 'a', encoding="utf-8") as file:
                count = 0

                while True:
                    try:
                        item = next(generator)
                    except StopIteration as e:
                        result = e.value
                        break

                    json.dump(asdict(item), file, ensure_ascii=False)
                    file.write('\n')

                    count += 1
                    print(f"{count} - готов!")

            return result                    
        return wrapper
    return decorator
            
        
@writeOutFile()
def generationQuestion(
    limit: int = 3
) -> Generator[QuestionDomain, None, None]:
    gsroffset, count = 0, 0

    while count < limit:
        article = wiki(gsroffset=gsroffset)
        
        gsroffset = article['continue']['gsroffset']
        print(f"gsroffset={gsroffset}")
        page = next(
            iter(article["query"]["pages"].values())
        )
        
        genQ = questionFromLLM(page['extract'])
        if genQ.success:
            qDom = QuestionDomain(
                id          = f"{SOURCE}-{count}",
                source      = SOURCE,
                page        = page['title'],
                url         = page['canonicalurl'],
                question    = genQ.question,
                answer      = genQ.answer,
                evidence    = genQ.evidence
            )
            
            count += 1
            
            yield qDom
        else:
            print(json.dumps(page, indent=4, ensure_ascii=False))
            print(genQ)

@writeOutFile(LOG_FILE)
def calcStat(
    logList: List[LogModelDomain]
) -> Generator[LogModelDomain, None, Tuple[int, float]]:
    right_count = 0
    total_cost = 0
    
    for elem in logList:
        if elem.right:
            right_count += 1

        total_cost += elem.cost
        
        yield elem

    return right_count, total_cost

def pipeline(
    f_gen_q : bool = True,
    f_gen_a : bool = True,
    model   : str  = STRONG_MODEL,
) -> None:
    if f_gen_q:
        print(f"Началась генерация вопросов. Результаты в {DATASET_FILE_NAME}")
        generationQuestion(limit = 40)

    if f_gen_a:
        print(f"Началась генерация ответов моделью {model}. Логи в {LOG_FILE}")
        logList: List[LogModelDomain] = list()
        modelSolution(model, logList)

        print(f"Начался подсчет правильных ответов")
        right_count, total_cost = calcStat(logList)
        print(f"right_count = {right_count}, total_cost = {total_cost}")


if __name__ == "__main__":
    pipeline(False, True, WEAK_MODEL)