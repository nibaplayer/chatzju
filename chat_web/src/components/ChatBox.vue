<template>
  <div>
    <div class="chat-window">
      <div v-for="(msg, index) in messages" :key="index" :class="[msg.sender,'message-box']"> 
        <!-- 绑定两个类型 -->
        <!-- class="msg.sender" msg时messages中的元素，其sender属性有两种：user与llm -->
        <span>{{msg.sender + ':' }}</span><br>
        
        <span>{{msg.text }}</span>
      </div>
    </div>
    <div class="container">
      <div class="input-container">
        <input
          v-model="input"
          @keydown.enter="handleSend"
          style="width: 90%; box-sizing: border-box;"
          placeholder="Type your message"
        />
        <button @click="handleSend">Send</button>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      messages: [],
      input: ''
    };
  },
  methods: {
    async handleSend() {
      if (this.input.trim()) {
        const userMessage = { sender: 'user', text: this.input };
        this.messages.push(userMessage);// 将用户消息记录到消息列表中
        
        // 发送请求到后端服务
        try {
          const response = await axios.post('http://10.214.149.209:14008/chatgraph', { message: this.input });
          // this.input = '~~~等待响应~~~';
          // console.log('Response:', response)
          const llmMessage = { sender: 'llm', text: response.data };
          // console.log('response.data.text', response.data)
          this.messages.push(llmMessage);
          // TODO 跳转到指定地址
          console.log("response.data", response.data)
          
          if(response.data['action']=='answer'){
            console.log(response.data['content'])
          }else if(response.data['action']=='open'){
            let destination = response.data['content']
            window.open(destination) // 在新窗口打开
          }
        } catch (error) {
          console.error('Error fetching response:', error);
        }
        
        this.input = '';
      }
    }
  }
};
</script>

<style>
.chat-window {
  border: 1px solid #ccc;
  padding: 10px;
  height: 400px;
  overflow-y: scroll;
}
.user {
  text-align: right;
  margin: 5px 0;
}
.llm {
  text-align: left;
  margin: 5px 0;
}
.message-box {
  border: 1px solid #ccc;
  padding: 10px;
  margin: 10px 0;
  border-radius: 5px;
}
.container{
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
}
.input-container {
  display: flex;
  align-items: center;
}

</style>
  