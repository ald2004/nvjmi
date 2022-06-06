#include <iostream>
#include <string>


#include "nvjmi.h"
#include "log/logging.h"
#ifdef _WIN32
//Windows
extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
#include "libavutil/imgutils.h"
};
#else
// linux
#ifdef __cplusplus
extern "C"
{
#endif
// #include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
// #include <libswscale/swscale.h>
// #include <libavutil/imgutils.h>
#ifdef __cplusplus
};
#endif
#endif
int main(){
    AVFormatContext *pFormatCtx;
    AVPacket *packet;
    av_register_all();
    avformat_network_init();
    pFormatCtx = avformat_alloc_context();
    const char *filepath = "/usr/src/jetson_multimedia_api/data/Video/sample_outdoor_car_1080p_10fps.h264";
    if (avformat_open_input(&pFormatCtx, filepath, NULL, NULL) != 0) {
        LogInfo("Couldn't open input stream.\n");
        return -1;
    }
    else
    {
        // std::cout<<"the h264 file :"<<filepath <<" is opened!"<<std::endl;
        LogInfo( "Input file read complete[%s]\n",filepath);
    }
    if (avformat_find_stream_info(pFormatCtx, NULL) < 0) {
        LogInfo("Couldn't find stream information.\n");
    }

    int height = 1080;
    int width = 1920;
    packet = (AVPacket *) av_malloc(sizeof(AVPacket));
    av_dump_format(pFormatCtx, 0, filepath, 0);

    /* Set thread name for decoder Output Plane thread. */
    pthread_setname_np(pthread_self(), "DecOutPlane");
    

    boe::nvJmiCtxParam jmi_ctx_param;
    if (width > 0 && height > 0) {
        jmi_ctx_param.resize_width = width;
        jmi_ctx_param.resize_height = height;
    }
    jmi_ctx_param.coding_type = boe::NV_VIDEO_CodingH264;
    std::string dec_name = "boedec";
    boe::nvJmiCtx *jmi_ctx_ ;
    jmi_ctx_= boe::nvjmi_create_decoder(dec_name.data(), &jmi_ctx_param);
    boe::nvPacket  nvpacket;
    while (av_read_frame(pFormatCtx, packet) >= 0) {
        nvpacket.payload_size = packet->size;
        nvpacket.payload = packet->data;
        int ret;
        ret = boe::nvjmi_decoder_put_packet(jmi_ctx_, &nvpacket);
        if(ret == boe::NVJMI_ERROR_STOP) {
            LogError("frameCallback: nvjmi decode error, frame callback EOF!");
        }

        
        while (ret >= 0) {
            boe::nvFrameMeta nvframe_meta;
            ret = boe::nvjmi_decoder_get_frame_meta(jmi_ctx_, &nvframe_meta);
            // std::cout << "+++++++++++++["<<ret<<"]++++++++++++++++++"<<std::endl;
            
            
            // Buffer buf;
            // buf.allocate(nvframe_meta.width, nvframe_meta.height, 3, nvframe_meta.payload_size / nvframe_meta.height);
            // jmi::nvjmi_decoder_retrieve_frame_data(jmi_ctx_, &nvframe_meta, (void*)buf.getData());   ctx_->frames;
            std::cout<< "got meta: \n coded_height["<<
              nvframe_meta.coded_height << "] coded_width[" <<
              nvframe_meta.coded_width <<"] frame_index[" <<
              nvframe_meta.frame_index <<"] height[" <<
              nvframe_meta.height <<"] width[" <<
              nvframe_meta.width <<"] payload_size[" <<
              nvframe_meta.payload_size <<"] timestamp[" <<
              nvframe_meta.timestamp << "]." <<std::endl; 
              if (ret < 0) break;
        
        
            // jmi::nvFrameMeta nvframe_meta;
            // ret = jmi::nvjmi_decoder_get_frame_meta(jmi_ctx_, &nvframe_meta);
            // if(ret == jmi::NVJMI_ERROR_FRAMES_EMPTY){
            //     std::cout<<"empty frame!!!\n";
            //     continue;
            // }
            // if (ret < 0) {
            //     LogInfo("now we got errors :[%d]\n",ret);
            //     std::cout<< "now nvframe_meta.height is: ["<<nvframe_meta.height<<"]"<<std::endl
            //     <<"nvframe_meta.width is: ["<<nvframe_meta.width<<"] \n"<<std::fflush;
            //     break;
            // };
            // // unsigned char* buf= new unsigned char[nvframe_meta.width, nvframe_meta.height, 3, nvframe_meta.payload_size / nvframe_meta.height];
            unsigned char* buf=(unsigned char*)malloc(1920* 1080*3*10);
            boe::nvjmi_decoder_retrieve_frame_data(jmi_ctx_, &nvframe_meta, (void*)buf);  
            // for(int p=0;p<1000;p++)
            //     std::cout<< (int)buf[p] <<" ";
            // std::cout<<std::endl; 
            free(buf);
            // LogInfo("now we get the packet size is:%d \n",packet->size);
            // break;
        }
    }
    if (packet != nullptr) {
        av_free(packet);
    }
}