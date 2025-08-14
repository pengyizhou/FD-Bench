class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.isRecording = false;
        
        this.port.onmessage = (event) => {
            if (event.data.command === 'setRecording') {
                this.isRecording = event.data.value;
            }
        };
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        const inputChannel = input[0];
        
        if (this.isRecording && inputChannel) {
            const int16Array = new Int16Array(inputChannel.length);
            for (let i = 0; i < inputChannel.length; i++) {
                int16Array[i] = inputChannel[i] * 0x7FFF;
            }
            
            this.port.postMessage({
                audio: Array.from(new Uint8Array(int16Array.buffer)),
                inputData: Array.from(inputChannel)
            });
        }
        
        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor); 