#三种方式：3.jsonschema 结构化输出 三种输出在底层都会转换成工具来给大模型处理


from my_llm import deepseek_llm
#创建json_schema

json_schema={
    "title":"movie",#工具名字叫做title,用法是电影标题 ...
    "description":"电影详细信息,包括标题，上映年份，导演与评分",
    "type":"object",
    "properties":{
        "title":{"type":"string","description":"电影标题"},
        "year":{"type":"integer","description":"电影上映年份"},
        "director":{"type":"string","description":"电影导演的中文名字"},
        "rating":{"type":"number","description":"电影评分"}
    },
    "required":["title","year","director","rating"]#指定映射字段时必须返回这些项目
    #json_schema必须包含以上5种固定
    #适用于动态传入时

}

model_with_structured_output=deepseek_llm.with_structured_output(json_schema)#后跟类名


resp=model_with_structured_output.invoke("介绍一下电影《闪灵》")
print(type(resp))
print(resp)
