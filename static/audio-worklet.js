// audio-worklet.js - Modern AudioWorklet processor for perfect audio handling

class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferSize = 1024;
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
        this.sampleRate = 16000;
        this.vadThreshold = 0.01;
        this.isRecording = false;
        
        // Voice Activity Detection state
        this.energyHistory = new Array(10).fill(0);
        this.energyIndex = 0;
        
        this.port.onmessage = (event) => {
            if (event.data.command === 'start') {
                this.isRecording = true;
            } else if (event.data.command === 'stop') {
                this.isRecording = false;
            }
        };
    }
    
    process(inputs, outputs, parameters) {
        const input = inputs[0];
        
        if (input.length > 0 && this.isRecording) {
            const inputData = input[0];
            
            // Calculate energy for VAD
            let energy = 0;
            for (let i = 0; i < inputData.length; i++) {
                energy += inputData[i] * inputData[i];
            }
            energy = Math.sqrt(energy / inputData.length);
            
            // Update energy history for smoothing
            this.energyHistory[this.energyIndex] = energy;
            this.energyIndex = (this.energyIndex + 1) % this.energyHistory.length;
            
            const avgEnergy = this.energyHistory.reduce((a, b) => a + b) / this.energyHistory.length;
            
            // Process audio in chunks
            for (let i = 0; i < inputData.length; i++) {
                this.buffer[this.bufferIndex] = inputData[i];
                this.bufferIndex++;
                
                if (this.bufferIndex >= this.bufferSize) {
                    // Send buffer to main thread with VAD info
                    this.port.postMessage({
                        type: 'audioData',
                        data: Array.from(this.buffer),
                        energy: avgEnergy,
                        vadActive: avgEnergy > this.vadThreshold
                    });
                    
                    this.bufferIndex = 0;
                }
            }
        }
        
        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor);