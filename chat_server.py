import os
#设置环境变量


from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from graph.navigation import navigator


#接入flask
app = Flask(__name__)
CORS(app)


# @app.route('/chat', methods = ['POST']) #弃用
# def chat():
#     # 未设置历史记录
#     print('get http request for record!')
#     message = request.json['message']
#     # print('message:',message)
#     action = action_chain(message)
#     # print('action:',action)
#     result = {'action':action}
#     if action == 'open':
#         answer = navigate_chain(message)
#         print('answer:',answer)
#         result['answer'] = answer
#         # 添加验证步骤
#     return jsonify(result)

@app.route('/chatgraph', methods = ['POST'])
def chatgraph():
    print('get http request for record!')
    message = request.json['message']
    llm_result = navigator.invoke({"question": message}) 
    return_result = llm_result['answer'] 
    # llm_result['answer'] 是一个json {'action':'','content':'url'}

    return jsonify(return_result)


    

if __name__ == '__main__':

    app.run(debug=True,host='0.0.0.0',port=14008,threaded=True)
