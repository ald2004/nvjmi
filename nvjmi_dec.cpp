#include "nvjmi.h"
#include "./include/NvApplicationProfiler.h"
#include "cuda_utils/cudaMappedMemory.h"
#include "logging.h"

#include "converter/vic_onverter.h"
#include "converter/cuda_converter.h"
#include "converter/fd_egl_frame_map.h"
#include "NvVideoDecoder.h"
#include "nvbuf_utils.h"
#include "NvBufSurface.h"
#include <tbb/concurrent_queue.h>

#include <vector>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <queue>
#include <atomic>
#include <fcntl.h>

using namespace std;


namespace boe {

// #define CHUNK_SIZE 1<<22
#define CHUNK_SIZE 4000000
#define MAX_BUFFERS 32
#define NAMELEN 16
 /**
 * LOG_NVJMI_DECODER logging prefix
 * @ingroup codec
  */
#define LOG_NVJMI_DECODER "[nvjmi-decoder] "

#define TEST_ERROR(condition, message, errorCode)    \
    if (condition)  {                              \
          LogError(LOG_NVJMI_DECODER "%s - %d\n", message, errorCode);     \
     }
#define TEST_ERRORX(cond, str, label) \
    if(cond) { \
        std::cerr << str << endl; \
        error = 1; \
        goto label; }

    struct nvJmiCtx {
        NvVideoDecoder *dec{};
        atomic<bool> eos{}; //流接入结束
        atomic<bool> output_plane_stop {}; //停止output_plane
        atomic<bool> capture_plane_stop{}; //停止capture_plane
        atomic<int> capture_plane_error_code{}; //capture plane错误
        bool got_res_event{};
        int index{};
        unsigned int coded_width{}; //编码的帧图像宽度
        unsigned int coded_height{}; //编码的帧图像高度
        unsigned int resize_width{};
        unsigned int resize_height{};
        unsigned int frame_size{};
        int dst_dma_fd{-1};
        int numberCaptureBuffers{};
        int dmaBufferFileDescriptor[MAX_BUFFERS]{};
        unsigned int decoder_pixfmt{};
        int blocking_mode;
        int max_perf;
        int extra_cap_plane_buffer;

        uint32_t video_height;
        uint32_t video_width;
        uint32_t display_height;
        uint32_t display_width;
        
        bool stats;
        bool got_error;
        bool enable_metadata;
        bool enable_input_metadata;

        int numCapBuffers;
        int loop_count;
        int dmabuff_fd[MAX_BUFFERS];

        std::thread * dec_capture_thread{};
        std::thread * output_plane_stop_thread{};
        pthread_t   dec_pollthread; // Polling thread, created if running in non-blocking mode.
        pthread_t dec_capture_loop; // Decoder capture thread, created if running in blocking mode.

        unsigned char * frame_buffer[MAX_BUFFERS]{};
        tbb::concurrent_bounded_queue<int> * frame_pools{};
        tbb::concurrent_bounded_queue<int> * frames{};

        unsigned long long timestamp[MAX_BUFFERS]{};

        enum v4l2_memory output_plane_mem_type;
        enum v4l2_memory capture_plane_mem_type;
        enum v4l2_skip_frames_type skip_frames;

        std::queue < NvBuffer * > *conv_output_plane_buf_queue;
        pthread_mutex_t queue_lock;
        pthread_cond_t queue_cond;  
        int stress_test;
        //for converter
        VICConverter * vic_converter{};
        CUDAConverter * cuda_converter{};
        FdEglFrameMap * fd_egl_frame_map{};

        cudaStream_t * cuda_stream;
    };

    /**
     * Exit on error.
     *
     * @param ctx : Decoder context
     */
    static void abort(nvJmiCtx *ctx){
        ctx->got_error = true;
        ctx->dec->abort();
    }

    /**
  * Query and Set Capture plane.
  *
  * @param ctx : Decoder context
  */
    static void
    query_and_set_capture(nvJmiCtx * ctx)
    {
        NvVideoDecoder *dec = ctx->dec;
        struct v4l2_format format;
        struct v4l2_crop crop;
        int32_t min_dec_capture_buffers;
        int ret = 0;
        int error = 0;
        uint32_t window_width;
        uint32_t window_height;
        uint32_t sar_width;
        uint32_t sar_height;
        
        NvBufSurfaceColorFormat pix_format;
        NvBufSurf::NvCommonAllocateParams params;
        NvBufSurf::NvCommonAllocateParams capParams;

        /* Get capture plane format from the decoder.
        This may change after resolution change event.
        Refer ioctl VIDIOC_G_FMT */
        ret = dec->capture_plane.getFormat(format);
        TEST_ERROR(ret < 0,
                "Error: Could not get format from decoder capture plane", error);

        /* Get the display resolution from the decoder.
        Refer ioctl VIDIOC_G_CROP */
        ret = dec->capture_plane.getCrop(crop);
        TEST_ERROR(ret < 0,
                "Error: Could not get crop from decoder capture plane", error);
        ctx->coded_width = crop.c.width;
        ctx->coded_height = crop.c.height;

        if (ctx->resize_width == 0 || ctx->resize_height == 0){
            ctx->resize_width = ctx->coded_width;
            ctx->resize_height = ctx->coded_height;
        }
        cout << "Video Resolution: " << crop.c.width << "x" << crop.c.height
            << endl;
        ctx->display_height = crop.c.height;
        ctx->display_width = crop.c.width;

        /* Get the Sample Aspect Ratio (SAR) width and height */
        ret = dec->getSAR(sar_width, sar_height);
        cout << "Video SAR width: " << sar_width << " SAR height: " << sar_height << endl;
        if(ctx->dst_dma_fd != -1)
        {
            ret = NvBufSurf::NvDestroy(ctx->dst_dma_fd);
            ctx->dst_dma_fd = -1;
            TEST_ERROR(ret < 0, "Error: Error in BufferDestroy", error);
        }
        /* Create PitchLinear output buffer for transform. */
        // params.memType = NVBUF_MEM_SURFACE_ARRAY;
        // params.width = crop.c.width;
        // params.height = crop.c.height;
        // params.layout = NVBUF_LAYOUT_PITCH;
        // if (ctx->out_pixfmt == 1)
        // params.colorFormat = NVBUF_COLOR_FORMAT_NV12;
        // else if (ctx->out_pixfmt == 2)
        // params.colorFormat = NVBUF_COLOR_FORMAT_YUV420;
        // else if (ctx->out_pixfmt == 3)
        // params.colorFormat = NVBUF_COLOR_FORMAT_NV16;
        // else if (ctx->out_pixfmt == 4)
        // params.colorFormat = NVBUF_COLOR_FORMAT_NV24;

        // params.memtag = NvBufSurfaceTag_VIDEO_CONVERT;

        // ret = NvBufSurf::NvAllocate(&params, 1, &ctx->dst_dma_fd);
        // TEST_ERROR(ret == -1, "create dmabuf failed", error);

        /* deinitPlane unmaps the buffers and calls REQBUFS with count 0 */
        dec->capture_plane.deinitPlane();
        if(1||ctx->capture_plane_mem_type == V4L2_MEMORY_DMABUF){
            for(int index = 0 ; index < ctx->numCapBuffers ; index++){
                if(ctx->dmabuff_fd[index] != 0){
                    ret = NvBufSurf::NvDestroy(ctx->dmabuff_fd[index]);
                    TEST_ERROR(ret < 0, "Error: Error in BufferDestroy", error);
                }
            }
        }
       
        ctx->vic_converter->exit();
        ctx->fd_egl_frame_map->exit();
        
        // ctx->frame_pools->clear();
        ctx->frames->clear();

        /* Not necessary to call VIDIOC_S_FMT on decoder capture plane.
        But decoder setCapturePlaneFormat function updates the class variables */
        ret = dec->setCapturePlaneFormat(format.fmt.pix_mp.pixelformat,
                                        format.fmt.pix_mp.width,
                                        format.fmt.pix_mp.height);
        TEST_ERROR(ret < 0, "Error in setting decoder capture plane format", error);

        ctx->video_height = format.fmt.pix_mp.height;
        ctx->video_width = format.fmt.pix_mp.width;
        /* Get the minimum buffers which have to be requested on the capture plane. */
        ret = dec->getMinimumCapturePlaneBuffers(min_dec_capture_buffers);
        
        TEST_ERROR(ret < 0,"Error while getting value of minimum capture plane buffers",error);
        // ctx->numberCaptureBuffers = min_dec_capture_buffers + 5;
        /* Request (min + extra) buffers, export and map buffers. */
        if(ctx->capture_plane_mem_type == V4L2_MEMORY_MMAP)
        {
            /* Request, Query and export decoder capture plane buffers.
            Refer ioctl VIDIOC_REQBUFS, VIDIOC_QUERYBUF and VIDIOC_EXPBUF */
            ret =
                dec->capture_plane.setupPlane(V4L2_MEMORY_MMAP,
                                            min_dec_capture_buffers + ctx->extra_cap_plane_buffer, false,
                                            false);
            TEST_ERROR(ret < 0, "Error in decoder capture plane setup", error);
        }
        else if(ctx->capture_plane_mem_type == V4L2_MEMORY_DMABUF)
        {
            /* Set colorformats for relevant colorspaces. */
            switch(format.fmt.pix_mp.colorspace)
            {
                case V4L2_COLORSPACE_SMPTE170M:
                    if (format.fmt.pix_mp.quantization == V4L2_QUANTIZATION_DEFAULT)
                    {
                        cout << "Decoder colorspace ITU-R BT.601 with standard range luma (16-235)" << endl;
                        pix_format = NVBUF_COLOR_FORMAT_NV12;
                    }
                    else
                    {
                        cout << "Decoder colorspace ITU-R BT.601 with extended range luma (0-255)" << endl;
                        pix_format = NVBUF_COLOR_FORMAT_NV12_ER;
                    }
                    break;
                case V4L2_COLORSPACE_REC709:
                    if (format.fmt.pix_mp.quantization == V4L2_QUANTIZATION_DEFAULT)
                    {
                        cout << "Decoder colorspace ITU-R BT.709 with standard range luma (16-235)" << endl;
                        pix_format =  NVBUF_COLOR_FORMAT_NV12_709;
                    }
                    else
                    {
                        cout << "Decoder colorspace ITU-R BT.709 with extended range luma (0-255)" << endl;
                        pix_format = NVBUF_COLOR_FORMAT_NV12_709_ER;
                    }
                    break;
                case V4L2_COLORSPACE_BT2020:
                    {
                        cout << "Decoder colorspace ITU-R BT.2020" << endl;
                        pix_format = NVBUF_COLOR_FORMAT_NV12_2020;
                    }
                    break;
                default:
                    cout << "supported colorspace details not available, use default" << endl;
                    if (format.fmt.pix_mp.quantization == V4L2_QUANTIZATION_DEFAULT)
                    {
                        cout << "Decoder colorspace ITU-R BT.601 with standard range luma (16-235)" << endl;
                        pix_format = NVBUF_COLOR_FORMAT_NV12;
                    }
                    else
                    {
                        cout << "Decoder colorspace ITU-R BT.601 with extended range luma (0-255)" << endl;
                        pix_format = NVBUF_COLOR_FORMAT_NV12_ER;
                    }
                    break;
            }

            ctx->numCapBuffers = min_dec_capture_buffers + ctx->extra_cap_plane_buffer;

            capParams.memType = NVBUF_MEM_SURFACE_ARRAY;
            capParams.width = crop.c.width;
            capParams.height = crop.c.height;
            capParams.layout = NVBUF_LAYOUT_BLOCK_LINEAR;
            capParams.memtag = NvBufSurfaceTag_VIDEO_DEC;

            if (format.fmt.pix_mp.pixelformat  == V4L2_PIX_FMT_NV24M)
            pix_format = NVBUF_COLOR_FORMAT_NV24;
            else if (format.fmt.pix_mp.pixelformat  == V4L2_PIX_FMT_NV24_10LE)
            pix_format = NVBUF_COLOR_FORMAT_NV24_10LE;
            if (ctx->decoder_pixfmt == V4L2_PIX_FMT_MJPEG)
            {
                capParams.layout = NVBUF_LAYOUT_PITCH;
                if (format.fmt.pix_mp.pixelformat == V4L2_PIX_FMT_YUV422M)
                {
                    pix_format = NVBUF_COLOR_FORMAT_YUV422;
                }
                else
                {
                    pix_format = NVBUF_COLOR_FORMAT_YUV420;
                }
            }





            
            /** Specifies BT.601 colorspace - Y/CbCr 4:2:0 multi-planar. NVBUF_COLOR_FORMAT_NV12, */
            capParams.colorFormat = pix_format;
            
            // ret = NvBufferCreateEx(&ctx->dmaBufferFileDescriptor[idx], &capture_params);
            ret = NvBufSurf::NvAllocate(&capParams, ctx->numCapBuffers, ctx->dmabuff_fd);

            TEST_ERROR(ret < 0, "Failed to create buffers", error);
            /* Request buffers on decoder capture plane.
            Refer ioctl VIDIOC_REQBUFS */
            ret = dec->capture_plane.reqbufs(V4L2_MEMORY_DMABUF,ctx->numCapBuffers);
            TEST_ERROR(ret, "Error in request buffers on capture plane", error);
        }
        // ctx->frame_pools->set_capacity(ctx->numCapBuffers);
        ctx->frames->set_capacity(ctx->numCapBuffers);
        // for (int i = 0; i < ctx->numCapBuffers; ++i){
        //     ctx->frame_pools->push(i);
        // }

        NvBufferRect src_rect;
        src_rect.top = 0; //top和left均为0，则为scale；否则为crop
        src_rect.left = 0;
        src_rect.width = crop.c.width;
        src_rect.height = crop.c.height;

        NvBufferRect dst_rect;
        dst_rect.top = 0;
        dst_rect.left = 0;
        dst_rect.width = crop.c.width;
        dst_rect.height = crop.c.height;

        //create transform dst dma fd
        NvBufferCreateParams transform_dst_params{};
        transform_dst_params.payloadType = NvBufferPayload_SurfArray;
        transform_dst_params.width = crop.c.width;
        transform_dst_params.height = crop.c.height;
        if (ctx->resize_width > 0 && ctx->resize_height > 0) {
            transform_dst_params.width = ctx->resize_width;
            transform_dst_params.height = ctx->resize_height;

            dst_rect.width = ctx->resize_width;
            dst_rect.height = ctx->resize_height;
        }
        transform_dst_params.layout = NvBufferLayout_Pitch;
        transform_dst_params.colorFormat = NvBufferColorFormat_ABGR32;
        transform_dst_params.nvbuf_tag = NvBufferTag_VIDEO_CONVERT;
        ret = NvBufferCreateEx(&ctx->dst_dma_fd, &transform_dst_params);
        TEST_ERROR(ret == -1, "create dst_dmabuf failed", ret);

        ctx->vic_converter->init(src_rect, dst_rect);
        ctx->fd_egl_frame_map->init();

        /* Decoder capture plane STREAMON.
        Refer ioctl VIDIOC_STREAMON */
        ret = dec->capture_plane.setStreamStatus(true);
        TEST_ERROR(ret < 0, "Error in decoder capture plane streamon", error);

        /* Enqueue all the empty decoder capture plane buffers. */
        for (uint32_t i = 0; i < dec->capture_plane.getNumBuffers(); i++)
        {
            struct v4l2_buffer v4l2_buf;
            struct v4l2_plane planes[MAX_PLANES];

            memset(&v4l2_buf, 0, sizeof(v4l2_buf));
            memset(planes, 0, sizeof(planes));

            v4l2_buf.index = i;
            v4l2_buf.m.planes = planes;
            v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            // v4l2_buf.memory = ctx->capture_plane_mem_type;
            v4l2_buf.memory = V4L2_MEMORY_DMABUF; 
            v4l2_buf.m.planes[0].m.fd = ctx->dmabuff_fd[i];   
            // if(ctx->capture_plane_mem_type == V4L2_MEMORY_DMABUF)
                v4l2_buf.m.planes[0].m.fd = ctx->dmabuff_fd[i];
            ret = dec->capture_plane.qBuffer(v4l2_buf, NULL);
            TEST_ERROR(ret < 0, "Error Qing buffer at output plane", error);
        }
        cout << "Query and set capture successful" << endl;
        return;

    error:
        if (error)
        {
            abort(ctx);
            cerr << "Error in " << __func__ << endl;
        }
    }


    /**
     * Report decoder output metadata.
     *
     * @param ctx      : Decoder context
     * @param metadata : Pointer to decoder output metadata struct
     */
    static void report_metadata(nvJmiCtx *ctx, v4l2_ctrl_videodec_outputbuf_metadata *metadata){
        uint32_t frame_num = ctx->dec->capture_plane.getTotalDequeuedBuffers() - 1;

        cout << "Frame " << frame_num << endl;

        if (metadata->bValidFrameStatus)
        {
            if (ctx->decoder_pixfmt == V4L2_PIX_FMT_H264)
            {
                /* metadata for H264 input stream. */
                switch(metadata->CodecParams.H264DecParams.FrameType)
                {
                    case 0:
                        cout << "FrameType = B" << endl;
                        break;
                    case 1:
                        cout << "FrameType = P" << endl;
                        break;
                    case 2:
                        cout << "FrameType = I";
                        if (metadata->CodecParams.H264DecParams.dpbInfo.currentFrame.bIdrFrame)
                        {
                            cout << " (IDR)";
                        }
                        cout << endl;
                        break;
                }
                cout << "nActiveRefFrames = " << metadata->CodecParams.H264DecParams.dpbInfo.nActiveRefFrames << endl;
            }

            if (ctx->decoder_pixfmt == V4L2_PIX_FMT_H265)
            {
                /* metadata for HEVC input stream. */
                switch(metadata->CodecParams.HEVCDecParams.FrameType)
                {
                    case 0:
                        cout << "FrameType = B" << endl;
                        break;
                    case 1:
                        cout << "FrameType = P" << endl;
                        break;
                    case 2:
                        cout << "FrameType = I";
                        if (metadata->CodecParams.HEVCDecParams.dpbInfo.currentFrame.bIdrFrame)
                        {
                            cout << " (IDR)";
                        }
                        cout << endl;
                        break;
                }
                cout << "nActiveRefFrames = " << metadata->CodecParams.HEVCDecParams.dpbInfo.nActiveRefFrames << endl;
            }

            if (metadata->FrameDecStats.DecodeError)
            {
                /* decoder error status metadata. */
                v4l2_ctrl_videodec_statusmetadata *dec_stats =
                    &metadata->FrameDecStats;
                cout << "ErrorType="  << dec_stats->DecodeError << " Decoded MBs=" <<
                    dec_stats->DecodedMBs << " Concealed MBs=" <<
                    dec_stats->ConcealedMBs << endl;
            }
        }
        else
        {
            cout << "No valid metadata for frame" << endl;
        }
    }


    void *dec_capture_loop_fcn(void *arg){
        nvJmiCtx* ctx = (nvJmiCtx*)arg;
        NvVideoDecoder *dec = ctx->dec;
        struct v4l2_event ev;
        int ret;
        char threadname[NAMELEN];
        pthread_getname_np(pthread_self(),threadname,NAMELEN);
        cout << "Starting decoder capture loop thread["<<threadname<<"]" << endl;
        struct v4l2_format v4l2Format;
        struct v4l2_crop v4l2Crop;
        struct v4l2_event v4l2Event;
        int wait_count{};

        /* Need to wait for the first Resolution change event, so that
        the decoder knows the stream resolution and can allocate appropriate
        buffers when we call REQBUFS. */
        do
        {
            /* Refer ioctl VIDIOC_DQEVENT */
            ret = dec->dqEvent(ev, 50000);
            if (ret < 0)
            {
                if (errno == EAGAIN)
                {
                    cerr <<
                        "Timed out waiting for first V4L2_EVENT_RESOLUTION_CHANGE"
                        << endl;
                }
                else
                {
                    cerr << "Error in dequeueing decoder event" << endl;
                }
                abort(ctx);
                break;
            }
        }
        while ((ev.type != V4L2_EVENT_RESOLUTION_CHANGE) && !ctx->got_error);

        /* Received the resolution change event, now can do query_and_set_capture. */
        if (!ctx->got_error){
            std::cout<<"now we got error and query_and_set_capture start !!! \n";
            query_and_set_capture(ctx);
        }
        
        /* Exit on error or EOS which is signalled in main() */
        // while (!(ctx->dec->isInError() || ctx->capture_plane_stop))
        while (!(ctx->got_error || dec->isInError())){
            NvBuffer *dec_buffer{};

            /* Check for Resolution change again.
            Refer ioctl VIDIOC_DQEVENT */
            ret = dec->dqEvent(ev, false);
            if (ret == 0){
                switch (ev.type)
                    case V4L2_EVENT_RESOLUTION_CHANGE:
                        query_and_set_capture(ctx);
                        continue;
                }
        

            // if (!ctx->got_res_event) {
            //     ret = ctx->dec->dqEvent(v4l2Event, 1000);
            //     if (ret == 0) {
            //         switch (v4l2Event.type) {
            //         case V4L2_EVENT_RESOLUTION_CHANGE:
            //             respondToResolutionEvent(v4l2Format, v4l2Crop, ctx);
            //             continue;
            //         }
            //     }
            //     else{
            //         ++wait_count;
            //         if (wait_count > 10) {
            //             ctx->capture_plane_error_code = NVJMI_ERROR_CAPTURE_PLANE_DQEVENT;
            //             LogInfo(LOG_NVJMI_DECODER "dqEvent error: capture plane set stopped\n");
            //             break;
            //         }
            //         continue;
            //     }
            // }

            // while (!ctx->capture_plane_stop) 
            /* Decoder capture loop */
            while(!ctx->capture_plane_stop){
                struct v4l2_buffer v4l2_buf;
                struct v4l2_plane planes[MAX_PLANES];
                memset(&v4l2_buf, 0, sizeof(v4l2_buf));
                memset(planes, 0, sizeof(planes));
                v4l2_buf.m.planes = planes;
                /* Dequeue a filled buffer. */
                if (ctx->dec->capture_plane.dqBuffer(v4l2_buf, &dec_buffer, NULL, 0)){
                    if (errno == EAGAIN) {
                        if (ctx->output_plane_stop) {
                            ctx->capture_plane_stop = true;
                            LogInfo(LOG_NVJMI_DECODER "capture plane set stopped\n");
                        }
                        usleep(1000);
                        if (v4l2_buf.flags & V4L2_BUF_FLAG_LAST){
                            cout << "Got EoS at capture plane" << endl;
                            goto handle_eos;
                        }
                        usleep(1000);
                    }
                    else {
                        // TEST_ERROR(errno != 0, "Error while calling dequeue at capture plane", errno);
                        // ctx->capture_plane_stop = true;
                        abort(ctx);
                        cerr << "Error while calling dequeue at capture plane" <<endl;
                    }
                    break;
                }
                
                if (ctx->enable_metadata){
                    v4l2_ctrl_videodec_outputbuf_metadata dec_metadata;

                    /* Get the decoder output metadata on capture-plane.
                    Refer V4L2_CID_MPEG_VIDEODEC_METADATA */
                    ret = dec->getMetadata(v4l2_buf.index, dec_metadata);
                    if (ret == 0)
                    {
                        report_metadata(ctx, &dec_metadata);
                    }
                }
                
                // if (ctx->copy_timestamp 
                //     cout << "[" << v4l2_buf.index << "]" "dec capture plane dqB timestamp [" <<
                //     v4l2_buf.timestamp.tv_sec << "s" << v4l2_buf.timestamp.tv_usec << "us]" << endl;
                // }
                
                if(0){
                    // if write to output file .
                }else{
                    /* If not writing to file, Queue the buffer back once it has been used. */
                    // if(ctx->capture_plane_mem_type == V4L2_MEMORY_DMABUF)
                    v4l2_buf.m.planes[0].m.fd = ctx->dmabuff_fd[v4l2_buf.index];
                    if (dec->capture_plane.qBuffer(v4l2_buf, NULL) < 0){
                        abort(ctx);
                        cerr <<
                            "Error while queueing buffer at decoder capture plane"
                            << endl;
                        break;
                    }
                }
                if (ctx->capture_plane_mem_type == V4L2_MEMORY_DMABUF)
                dec_buffer->planes[0].fd = ctx->dmabuff_fd[v4l2_buf.index];
                    // dec_buffer->planes[0].fd = ctx->dmaBufferFileDescriptor[v4l2_buf.index];

                // do vic conversion conversion: color map convert (NV12@res#1 --> RGBA packed) and scale
                ret = ctx->vic_converter->convert(dec_buffer->planes[0].fd, ctx->dst_dma_fd);        
                TEST_ERROR(ret == -1, "Transform failed", ret);

                //get cuda pointer frm dma fd 1075
                // std::cout<<"mmmmmmmmmmmmmmmmmmmmmmmmmm " <<ctx->dst_dma_fd<<std::endl;
                cudaEglFrame egl_frame = ctx->fd_egl_frame_map->get(ctx->dst_dma_fd);

                int buf_index{ -1 };
                // std::cout << "dddddddddddddddddddddd ["<<ctx->frame_pools->size()<<"]" <<std::endl;

                while (!ctx->capture_plane_stop && !ctx->frame_pools->try_pop(buf_index)) {
                    // std::cout << "cccccccccccccccccccccccccccccc ["<<buf_index<<"]" <<std::endl;
                    std::this_thread::yield();
                }
                
                if (!ctx->frame_size){
                    // std::cout << "ddddddddddddddddddddddddddd ["<<ctx->frame_size<<"]" <<std::endl;//0
                    ctx->frame_size = (int)(ctx->resize_width*ctx->resize_height * 3 * sizeof(unsigned char));
                    // std::cout << "ddddddddddddddddddddddddddd ["<<ctx->frame_size<<"]" <<std::endl; //6220800

                    // ctx->frame_size =3110400;
                }

                if (!ctx->capture_plane_stop && (buf_index < MAX_BUFFERS && buf_index >= 0)) {
                    if (ctx->frame_buffer[buf_index] == nullptr){
                        if (!cudaAllocMapped((void**)&ctx->frame_buffer[buf_index], ctx->resize_width, ctx->resize_height, imageFormat::IMAGE_BGR8)) {
                            std::cout<< "hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh"<<std::endl;
                            break;
                        }
                    }

                    // do CUDA conversion: RGBA packed@res#2 --> BGR planar@res#2
                    
                    ctx->cuda_converter->convert(egl_frame,
                        ctx->resize_width,
                        ctx->resize_height,
                        COLOR_FORMAT_BGR,
                        (void *)ctx->frame_buffer[buf_index],
                        *ctx->cuda_stream);

                    cudaStreamSynchronize(*ctx->cuda_stream);

                //     ctx->timestamp[buf_index] = v4l2_buf.timestamp.tv_usec;
                    while (!ctx->capture_plane_stop && !ctx->frames->try_push(buf_index)) {
                        std::this_thread::yield();
                    }
                }
                else{
                    break;
                }

                // v4l2_buf.m.planes[0].m.fd = ctx->dmaBufferFileDescriptor[v4l2_buf.index];
                // if (ctx->dec->capture_plane.qBuffer(v4l2_buf, NULL) < 0){
                //     ERROR_MSG("Error while queueing buffer at decoder capture plane");
                // }
            }
        }
		
		// ctx->eos = true;
        // ctx->output_plane_stop = true;
        // ctx->capture_plane_stop = true;
        // ctx->dec->capture_plane.setStreamStatus(false);
        // LogInfo(LOG_NVJMI_DECODER "capture plane thread stopped\n");
    handle_eos:
        cout << "Exiting decoder capture loop thread" << endl;
        return NULL;
    }

    void *output_plane_stop_fcn(void *arg){
        nvJmiCtx* ctx = (nvJmiCtx*)arg;

        int ret{};
        while (!ctx->output_plane_stop && ctx->dec->output_plane.getNumQueuedBuffers() > 0 && !ctx->dec->isInError()) {
            struct v4l2_buffer v4l2_buf;
            struct v4l2_plane planes[MAX_PLANES];

            memset(&v4l2_buf, 0, sizeof(v4l2_buf));
            memset(planes, 0, sizeof(planes));

            v4l2_buf.m.planes = planes;
            ret = ctx->dec->output_plane.dqBuffer(v4l2_buf, NULL, NULL, -1);
            if (ret < 0) {
                TEST_ERROR(ret < 0, "Eos handling Error: DQing buffer at output plane", ret);
                break;
            }
        }

        ctx->output_plane_stop = true;

        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));

        v4l2_buf.m.planes = planes;
        v4l2_buf.m.planes[0].bytesused = 0;

        ret = ctx->dec->output_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0) {
            TEST_ERROR(ret < 0, "Error Qing buffer at output plane", ret);
        }
        LogInfo(LOG_NVJMI_DECODER "capture plane stopping ...\n");
    }
    
    /*
    * NVJMI API
    */
    JMI_API nvJmiCtx* nvjmi_create_decoder(char const* dec_name, nvJmiCtxParam* param) {
        int ret{};
        char*nalu_parse_buffer=NULL;
        log_level = DEFAULT_LOG_LEVEL;
        int error=0;
        cudaError_t err ;

        //create nvjmi context
        nvJmiCtx* ctx = new nvJmiCtx;
        ctx->resize_width = param->resize_width;
        ctx->resize_height = param->resize_height;
        ctx->stress_test = 1;
        ctx->blocking_mode=1;
        ctx->stats=1;
        ctx->max_perf=1;
        ctx->enable_input_metadata=true;
        ctx->output_plane_mem_type = V4L2_MEMORY_MMAP;
        ctx->capture_plane_mem_type = V4L2_MEMORY_DMABUF;
        ctx->enable_metadata=true;
        ctx->extra_cap_plane_buffer =5;
        ctx->enable_metadata=false;

        pthread_mutex_init(&ctx->queue_lock, NULL);
        pthread_cond_init(&ctx->queue_cond, NULL);
        //#define GOVERNOR_SYS_FILE "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        //#define REQUIRED_GOVERNOR "performance" "schedutil"
        NvApplicationProfiler &profiler = NvApplicationProfiler::getProfilerInstance();

        /* Create NvVideoDecoder object for blocking or non-blocking I/O mode. */
        if (ctx->blocking_mode)
        {
            cout << "Creating decoder in blocking mode \n";
            ctx->dec = NvVideoDecoder::createVideoDecoder("dec0");
        }
        else
        {
            cout << "Creating decoder in non-blocking mode \n";
            ctx->dec = NvVideoDecoder::createVideoDecoder("dec0", O_NONBLOCK);
        }
        TEST_ERRORX(!ctx->dec, "Could not create decoder", cleanup);

        /* Enable profiling for decoder if stats are requested. */
        if (ctx->stats){
            profiler.start(NvApplicationProfiler::DefaultSamplingInterval);
            ctx->dec->enableProfiling();
        }
        
        /* Subscribe to Resolution change event.
            Refer ioctl VIDIOC_SUBSCRIBE_EVENT */
        ret = ctx->dec->subscribeEvent(V4L2_EVENT_RESOLUTION_CHANGE, 0, 0);
        TEST_ERRORX(ret < 0, "Could not subscribe to V4L2_EVENT_RESOLUTION_CHANGE",cleanup);
        

        switch (param->coding_type) {
        case NV_VIDEO_CodingH264:
            ctx->decoder_pixfmt = V4L2_PIX_FMT_H264;
            break;
        case NV_VIDEO_CodingHEVC:
            ctx->decoder_pixfmt = V4L2_PIX_FMT_H265;
            break;
        case NV_VIDEO_CodingMPEG4:
            ctx->decoder_pixfmt = V4L2_PIX_FMT_MPEG4;
            break;
        case NV_VIDEO_CodingMPEG2:
            ctx->decoder_pixfmt = V4L2_PIX_FMT_MPEG2;
            break;
        case NV_VIDEO_CodingVP8:
            ctx->decoder_pixfmt = V4L2_PIX_FMT_VP8;
            break;
        case NV_VIDEO_CodingVP9:
            ctx->decoder_pixfmt = V4L2_PIX_FMT_VP9;
            break;
        default:
            ctx->decoder_pixfmt = V4L2_PIX_FMT_H264;
            break;
        }
        
        /* Set format on the output plane.
            Refer ioctl VIDIOC_S_FMT */
        ret = ctx->dec->setOutputPlaneFormat(ctx->decoder_pixfmt, CHUNK_SIZE);
        TEST_ERRORX(ret < 0, "Could not set output plane format", cleanup);
        
        /* Input to the decoder will be nal units. */
        nalu_parse_buffer = new char[CHUNK_SIZE];
        printf("Setting frame input mode to 0 \n");
        ret = ctx->dec->setFrameInputMode(0);
        TEST_ERRORX(ret < 0,"Error in decoder setFrameInputMode", cleanup);

        /* Disable decoder DPB management.
            NOTE: V4L2_CID_MPEG_VIDEO_DISABLE_DPB should be set after output plane
                    set format */
        ret = ctx->dec->disableDPB();
        TEST_ERRORX(ret < 0, "Error in decoder disableDPB", cleanup);        
   
        /* Enable decoder error and metadata reporting.
            Refer V4L2_CID_MPEG_VIDEO_ERROR_REPORTING */
        if (ctx->enable_metadata || ctx->enable_input_metadata){
            ret = ctx->dec->enableMetadataReporting();
            TEST_ERRORX(ret < 0, "Error while enabling metadata reporting", cleanup);
        }
        /* Enable max performance mode by using decoder max clock settings.
        Refer V4L2_CID_MPEG_VIDEO_MAX_PERFORMANCE */
        if (ctx->max_perf){
            ret = ctx->dec->setMaxPerfMode(ctx->max_perf);
            TEST_ERRORX(ret < 0, "Error while setting decoder to max perf", cleanup);
        }

        /* Set the skip frames property of the decoder.
        Refer V4L2_CID_MPEG_VIDEO_SKIP_FRAMES */
        if (ctx->skip_frames){
            ret = ctx->dec->setSkipFrames(ctx->skip_frames);
            TEST_ERRORX(ret < 0, "Error while setting skip frames param", cleanup);
        }

        /* Query, Export and Map the output plane buffers so can read
        encoded data into the buffers. */
        if (ctx->output_plane_mem_type == V4L2_MEMORY_MMAP) {
            /* configure decoder output plane for MMAP io-mode.
            Refer ioctl VIDIOC_REQBUFS, VIDIOC_QUERYBUF and VIDIOC_EXPBUF */
            ret = ctx->dec->output_plane.setupPlane(V4L2_MEMORY_MMAP, 1, true, false);
            // ret = ctx->dec->output_plane.setupPlane(V4L2_MEMORY_MMAP, 2, true, false);
        } else if (ctx->output_plane_mem_type == V4L2_MEMORY_USERPTR) {
            /* configure decoder output plane for USERPTR io-mode.
            Refer ioctl VIDIOC_REQBUFS */
            ret = ctx->dec->output_plane.setupPlane(V4L2_MEMORY_USERPTR, 10, false, true);
        }
        TEST_ERRORX(ret < 0, "Error while setting up output plane", cleanup);

        /* Start stream processing on decoder output-plane.
        Refer ioctl VIDIOC_STREAMON */
        ret = ctx->dec->output_plane.setStreamStatus(true);
        TEST_ERRORX(ret < 0, "Error in output plane stream on", cleanup);

        /* Enable copy timestamp with start timestamp in seconds for decode fps.
        NOTE: Used to demonstrate how timestamp can be associated with an
                individual H264/H265 frame to achieve video-synchronization. */
        // if (ctx->copy_timestamp && ctx->input_nalu) {
        //     ctx->timestamp = (ctx->start_ts * MICROSECOND_UNIT);
        //     ctx->timestampincr = (MICROSECOND_UNIT * 16) / ((uint32_t) (ctx->dec_fps * 16));
        // }

        // ctx->dec->output_plane.setStreamStatus(true);
        // TEST_ERROR(ret < 0, "Error in output plane stream on", ret);

        if(ctx->blocking_mode){
            ctx->dec_capture_thread = new thread(dec_capture_loop_fcn, ctx);
            /* Set thread name for decoder Capture Plane thread. */
            
        }
        

        ctx->frame_pools = new tbb::concurrent_bounded_queue<int>;
        ctx->frames = new tbb::concurrent_bounded_queue<int>;
        ctx->numberCaptureBuffers = 0;
        ctx->vic_converter = new VICConverter;
        ctx->cuda_converter = new CUDAConverter;
        ctx->fd_egl_frame_map = new FdEglFrameMap;

        //create cuda stream for cuda converter
        ctx->cuda_stream = new cudaStream_t;
        err = cudaStreamCreateWithFlags(ctx->cuda_stream, cudaStreamNonBlocking);
        // std::cout<<"bbbbbbbbbbbbbbbbbbbbbbbbbbbb"<<std::endl;
        if (err != cudaSuccess) {
            LogError(LOG_NVJMI_DECODER "cudaStreamCreateWithFlags: CUDA Runtime API error: %d - %s\n", (int)err, cudaGetErrorString(err));
            return nullptr;
        }

        //create frame buffer pools
        ctx->frame_pools->set_capacity(MAX_BUFFERS);
        ctx->frames->set_capacity(MAX_BUFFERS);
        ctx->frame_pools->clear();
        ctx->frames->clear();
        for (int i = 0; i < MAX_BUFFERS; ++i){
            ctx->frame_pools->push(i);
        }
        std::cout<< "init ctx complete! ["<<ctx->frame_pools->size()<<"]"<<std::endl;;
        return ctx;
cleanup:
        // if (ctx->blocking_mode && ctx->dec_capture_loop){
        //         pthread_join(ctx->dec_capture_loop, NULL);
        // }
        if (ctx->stats){
            profiler.stop();
            ctx->dec->printProfilingStats(cout);
            // if (ctx->renderer){
            //     ctx.renderer->printProfilingStats(cout);
            // }
            profiler.printProfilerData(cout);
        }

        if(ctx->capture_plane_mem_type == V4L2_MEMORY_DMABUF){
            for(int index = 0 ; index < ctx->numCapBuffers ; index++){
                if(ctx->dmabuff_fd[index] != 0)
                {
                    ret = NvBufSurf::NvDestroy(ctx->dmabuff_fd[index]);
                    if(ret < 0)
                    {
                        cerr << "Failed to Destroy NvBuffer" << endl;
                    }
                }
            }
        }
        if (ctx->dec && ctx->dec->isInError()){
            cerr << "Decoder is in error" << endl;
            error = 1;
        }
        if (ctx->got_error){
            error = 1;
        }

        /* The decoder destructor does all the cleanup i.e set streamoff on output and
        capture planes, unmap buffers, tell decoder to deallocate buffer (reqbufs
        ioctl with count = 0), and finally call v4l2_close on the fd. */
        delete ctx->dec;
        // /* Similarly, EglRenderer destructor does all the cleanup. */
        // delete ctx->renderer;
        // for (uint32_t i = 0 ; i < ctx.file_count ; i++)
        // delete ctx.in_file[i];
        // delete ctx.out_file;
        if(ctx->dst_dma_fd != -1){
            ret = NvBufSurf::NvDestroy(ctx->dst_dma_fd);
            ctx->dst_dma_fd = -1;
            if(ret < 0){
                cerr << "Error in BufferDestroy" << endl;
                error = 1;
            }
        }
        // delete[] nalu_parse_buffer;

        // free (ctx.in_file);
        // for (uint32_t i = 0 ; i < ctx.file_count ; i++)
        // free (ctx.in_file_path[i]);
        // free (ctx.in_file_path);
        // free(ctx.out_file_path);
        // if (!ctx->blocking_mode)
        // {
        //     sem_destroy(&ctx->pollthread_sema);
        //     sem_destroy(&ctx->decoderthread_sema);
        // }

        return NULL;
    }

    JMI_API int nvjmi_decoder_put_packet(nvJmiCtx* ctx, nvPacket* packet){
        // if (ctx->eos){
        //     if (ctx->capture_plane_stop) {
        //         return NVJMI_ERROR_STOP;			
		// 	}           
			
		// 	return NVJMI_ERROR_EOS;
        // }

        int ret{};

        if (packet->payload_size == 0){
            ctx->eos = true;
            LogInfo(LOG_NVJMI_DECODER "Input file read complete\n");

            ctx->output_plane_stop_thread = new thread(output_plane_stop_fcn, ctx);
            return NVJMI_OK;
        }
        
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];
        NvBuffer *nvBuffer;
        
        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, sizeof(planes));
        // nvBuffer = ctx->dec->output_plane.getNthBuffer(0);
        v4l2_buf.m.planes = planes;
        // v4l2_buf.index = 0;
        // nvBuffer->planes[0].bytesused=packet->payload_size;
        // std::cout<<"aaaaaaaaaaaaaaaaaaaaaaa     "<<packet->payload_size<<std::endl;
        // v4l2_buf.m.planes[0].bytesused = nvBuffer->planes[0].bytesused;
        
        


        if ((ctx->index < (int)ctx->dec->output_plane.getNumBuffers())) {
            nvBuffer = ctx->dec->output_plane.getNthBuffer(ctx->index);
        }
        else {
            ret = ctx->dec->output_plane.dqBuffer(v4l2_buf, &nvBuffer, NULL, -1);
            if (ret < 0) {
                TEST_ERROR(ret < 0, "Error DQing buffer at output plane", ret);
                return NVJMI_ERROR_OUTPUT_PLANE_DQBUF;
            }
        }
        
        memcpy(nvBuffer->planes[0].data, packet->payload, packet->payload_size);
        nvBuffer->planes[0].bytesused = packet->payload_size;

        if (ctx->index < ctx->dec->output_plane.getNumBuffers()) {
            v4l2_buf.index = ctx->index;
            v4l2_buf.m.planes = planes;
        }

        v4l2_buf.m.planes[0].bytesused = nvBuffer->planes[0].bytesused;

        v4l2_buf.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
        v4l2_buf.timestamp.tv_usec = packet->pts;// - (v4l2_buf.timestamp.tv_sec * (time_t)1000000);
        
        ret = ctx->dec->output_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0) {
            TEST_ERROR(ret < 0, "Error Qing buffer at output plane", ret);
            return NVJMI_ERROR_OUTPUT_PLANE_QBUF;
        }

        if (ctx->index < ctx->dec->output_plane.getNumBuffers())
            ctx->index++;
        
        
        return NVJMI_OK;
    }

    JMI_API int nvjmi_decoder_get_frame_meta(nvJmiCtx* ctx, nvFrameMeta* frame_meta) {
        int ret{};
        int frame_index{-1};
        // std::cout<<" ------------- ctx frames is: ["<<ctx->frames->size()<<"]"<<std::endl;
        if (ctx->dec->isInError()){
            return NVJMI_ERROR_DEC_INTERNAL;
        }

        if (ctx->capture_plane_error_code != NVJMI_OK){
            return ctx->capture_plane_error_code;
        }

        while (ctx->frames->try_pop(frame_index)){
            if (frame_index == -1) {
                return NVJMI_ERROR_GET_FRAME_META;
            }

            frame_meta->coded_width = ctx->coded_width;
            frame_meta->coded_height = ctx->coded_height;
            frame_meta->width = ctx->resize_width;
            frame_meta->height = ctx->resize_height;
            frame_meta->payload_size = ctx->frame_size;
            frame_meta->timestamp = ctx->timestamp[frame_index];
            frame_meta->frame_index = frame_index;
            frame_meta->got_data = 0;

            return ctx->frames->size();
        }

        if (ctx->capture_plane_stop) {
            return NVJMI_ERROR_STOP;
        }

        if (ctx->eos){
            return NVJMI_ERROR_EOS;
        }

        return NVJMI_ERROR_FRAMES_EMPTY;
    }

    JMI_API int nvjmi_decoder_retrieve_frame_data(nvJmiCtx* ctx, nvFrameMeta* frame_meta, void* frame_data){
        if (frame_data){
            if(ctx->frame_buffer[frame_meta->frame_index]==NULL){

            }else
            memcpy((unsigned char*)frame_data, ctx->frame_buffer[frame_meta->frame_index], frame_meta->payload_size);
            frame_meta->got_data = 1;
        }

        while (!ctx->frame_pools->try_push(frame_meta->frame_index) && !ctx->capture_plane_stop){
            std::this_thread::yield();
        }
        return NVJMI_OK;
    }

    JMI_API int nvjmi_decoder_close(nvJmiCtx* ctx){
        ctx->eos = true;
        ctx->output_plane_stop = true;
        ctx->capture_plane_stop = true;

        ctx->dec->abort();

        if (ctx->dec_capture_thread && ctx->dec_capture_thread->joinable()) {
            ctx->dec_capture_thread->join();
        }

        if (ctx->output_plane_stop_thread &&ctx->output_plane_stop_thread->joinable()) {
            ctx->output_plane_stop_thread->join();
        }

        LogInfo(LOG_NVJMI_DECODER "------>nvjmi_decoder_close\n");
        return NVJMI_OK;
    }

    JMI_API int nvjmi_decoder_free_context(nvJmiCtx** ctx) {
        auto& pctx = *ctx;

        if (pctx->dec_capture_thread) {
            delete pctx->dec_capture_thread;
            pctx->dec_capture_thread = nullptr;
        }

        if (pctx->output_plane_stop_thread) {
            if (pctx->output_plane_stop_thread->joinable()) {
                pctx->output_plane_stop_thread->join();
            }          
	    delete pctx->output_plane_stop_thread;
            pctx->output_plane_stop_thread = nullptr;
        }

        delete pctx->dec; pctx->dec = nullptr;

        if (pctx->dst_dma_fd != -1) {
            NvBufferDestroy(pctx->dst_dma_fd);
            pctx->dst_dma_fd = -1;
        }

        for (int idx = 0; idx < pctx->numberCaptureBuffers; idx++) {
            if (pctx->dmaBufferFileDescriptor[idx] != 0) {
                int ret = NvBufferDestroy(pctx->dmaBufferFileDescriptor[idx]);
                TEST_ERROR(ret < 0, "Failed to Destroy NvBuffer", ret);
            }
        }

        pctx->vic_converter->exit();
        pctx->fd_egl_frame_map->exit();

        for (int idx = 0; idx < MAX_BUFFERS; idx++){
            if (pctx->frame_buffer[idx]){
                cudaFreeHost(pctx->frame_buffer[idx]);
                pctx->frame_buffer[idx] = nullptr;
            }
        }

        delete pctx->frame_pools; pctx->frame_pools = nullptr;
        delete pctx->frames; pctx->frames = nullptr;

        delete pctx->vic_converter; pctx->vic_converter = nullptr;
        delete pctx->cuda_converter; pctx->cuda_converter = nullptr;
        delete pctx->fd_egl_frame_map; pctx->fd_egl_frame_map = nullptr;

        delete pctx; pctx = nullptr;

        LogInfo(LOG_NVJMI_DECODER "------>nvjmi_decoder_free_context!!!\n");

        return NVJMI_OK;
    }
}
