#include "arg.h"
#include "common.h"
#include "console.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "chat.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

static llama_context           ** g_ctx;
static llama_model             ** g_model;
static common_sampler          ** g_smpl;
static common_params            * g_params;
static std::vector<llama_token> * g_input_tokens;
static std::ostringstream       * g_output_ss;
static std::vector<llama_token> * g_output_tokens;
static bool is_interacting  = false;
static bool need_insert_eot = false;

static void print_usage(int argc, char ** argv) {
    (void) argc;

    LOG("\nexample usage:\n");
    LOG("\n  text generation:     %s -m your_model.gguf -p \"I believe the meaning of life is\" -n 128 -no-cnv\n", argv[0]);
    LOG("\n  chat (conversation): %s -m your_model.gguf -sys \"You are a helpful assistant\"\n", argv[0]);
    LOG("\n");
}

static bool file_exists(const std::string & path) {
    std::ifstream f(path.c_str());
    return f.good();
}

static bool file_is_empty(const std::string & path) {
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    return f.tellg() == 0;
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!is_interacting && g_params->interactive) {
            is_interacting  = true;
            need_insert_eot = true;
        } else {
            console::cleanup();
            LOG("\n");
            common_perf_print(*g_ctx, *g_smpl);

            // make sure all logs are flushed
            LOG("Interrupted by user\n");
            common_log_pause(common_log_main());

            _exit(130);
        }
    }
}
#endif

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    g_params = &params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMPLETION, print_usage)) {
        return 1;
    }

    auto & sparams = params.sampling;

    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    if (params.embedding) {
        LOG_ERR("************\n");
        LOG_ERR("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        LOG_ERR("************\n\n");

        return 0;
    }

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_WRN("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    if (params.rope_freq_base != 0.0) {
        LOG_WRN("%s: warning: changing RoPE frequency base to %g.\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 0.0) {
        LOG_WRN("%s: warning: scaling RoPE frequency by %g.\n", __func__, params.rope_freq_scale);
    }

    LOG_INF("%s: llama backend init\n", __func__);

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    common_sampler * smpl = nullptr;

    g_model = &model;
    g_ctx = &ctx;
    g_smpl = &smpl;

    std::vector<common_chat_msg> chat_msgs;

    // load the model and apply lora adapter, if any
    LOG_INF("%s: load the model and apply lora adapter, if any\n", __func__);

    auto llama_init = common_init_from_params(params);

    ctx   = llama_init->context();
    model = llama_init->model();
    smpl  = llama_init->sampler(0);

    if (ctx == NULL) {
        LOG_ERR("%s: error: unable to create context\n", __func__);
        return 1;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // note: the time for chat template initialization is not negligible:
    auto chat_templates = common_chat_templates_init(model, params.chat_template);

    // start measuring performance timings from here
    llama_perf_context_reset(ctx);

    LOG_INF("%s: llama threadpool init, n_threads = %d\n", __func__, (int) params.cpuparams.n_threads);

    // 为 llama_context 准备并挂接 CPU backend 的线程池，让后续 落在 CPU 上执行的 ggml 算子有并行执行环境。
    auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        LOG_ERR("%s: no CPU backend found\n", __func__);
        return 1;
    }
    auto * reg = ggml_backend_dev_backend_reg(cpu_dev);
    // 线程池实现属于 CPU backend 提供的能力
    auto * ggml_threadpool_new_fn = (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    auto * ggml_threadpool_free_fn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");

    // 根据两套 CPU 参数构造两套线程池配置: tpp_batch 和 tpp
    struct ggml_threadpool_params tpp_batch =   //  batch / prefill 阶段用一套线程数
            ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);
    struct ggml_threadpool_params tpp =         // 非 batch / 单 token decode 阶段用另一套线程数
            ggml_threadpool_params_from_cpu_params(params.cpuparams);

    if (!set_process_priority(params.cpuparams.priority)) {
        LOG_ERR("%s: error: failed to set process priority\n", __func__);
        return 1;
    }

    // prefill / prompt 批量处理：通常吞吐优先，适合更多线程，对应 threadpool_batch
    // decode / 单 token 生成：通常时延优先，可能用另一套线程数，对应 threadpool
    struct ggml_threadpool * threadpool_batch = NULL;
    if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {  // 如果两套参数不同，就额外创建一个 batch 线程池
        // 后面运行时会在两套线程池之间切换，而不是一套池子硬撑两种负载
        threadpool_batch = ggml_threadpool_new_fn(&tpp_batch);
        if (!threadpool_batch) {
            LOG_ERR("%s: batch threadpool create failed : n_threads %d\n", __func__, tpp_batch.n_threads);
            return 1;
        }

        // start the non-batch threadpool in the paused state
        tpp.paused = true;
    }

    // 创建普通线程池并挂到 context
    struct ggml_threadpool * threadpool = ggml_threadpool_new_fn(&tpp);
    if (!threadpool) {
        LOG_ERR("%s: threadpool create failed : n_threads %d\n", __func__, tpp.n_threads);
        return 1;
    }

    // 挂完之后，ctx 在真正执行图时就能按场景选用合适的 CPU 线程池
    llama_attach_threadpool(ctx, threadpool, threadpool_batch);

    const int n_ctx_train = llama_model_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);

    if (n_ctx > n_ctx_train) {
        LOG_WRN("%s: model was trained on only %d context tokens (%d specified)\n", __func__, n_ctx_train, n_ctx);
    }

    // auto enable conversation mode if chat template is available
    const bool has_chat_template = common_chat_templates_was_explicit(chat_templates.get());
    if (params.conversation_mode == COMMON_CONVERSATION_MODE_AUTO) {
        if (has_chat_template) {
            LOG_INF("%s: chat template is available, enabling conversation mode (disable it with -no-cnv)\n", __func__);
            params.conversation_mode = COMMON_CONVERSATION_MODE_ENABLED;
        } else {
            params.conversation_mode = COMMON_CONVERSATION_MODE_DISABLED;
        }
    }

    // in case user force-activate conversation mode (via -cnv) without proper chat template, we show a warning
    if (params.conversation_mode && !has_chat_template) {
        LOG_WRN("%s: chat template is not available or is not supported. This may cause the model to output suboptimal responses\n", __func__);
    }

    // print chat template example in conversation mode
    if (params.conversation_mode) {
        if (params.enable_chat_template) {
            if (!params.prompt.empty() && params.system_prompt.empty()) {
                LOG_WRN("*** User-specified prompt will pre-start conversation, did you mean to set --system-prompt (-sys) instead?\n");
            }

            LOG_INF("%s: chat template example:\n%s\n", __func__, common_chat_format_example(chat_templates.get(), params.use_jinja, params.default_template_kwargs).c_str());
        } else {
            LOG_INF("%s: in-suffix/prefix is specified, chat template will be disabled\n", __func__);
        }
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    std::string path_session = params.path_prompt_cache;
    std::vector<llama_token> session_tokens;

    if (!path_session.empty()) {
        LOG_INF("%s: attempting to load saved session from '%s'\n", __func__, path_session.c_str());
        if (!file_exists(path_session)) {
            LOG_INF("%s: session file does not exist, will create.\n", __func__);
        } else if (file_is_empty(path_session)) {
            LOG_INF("%s: The session file is empty. A new session will be initialized.\n", __func__);
        } else {
            // The file exists and is not empty
            session_tokens.resize(n_ctx);
            size_t n_token_count_out = 0;
            // TODO：这里的函数需要进一步分析
            // 加载 token ID作为输入，将 token 对应的 KV cache 加载到 host 或者 device 的 memory 中
            if (!llama_state_load_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                LOG_ERR("%s: failed to load session file '%s'\n", __func__, path_session.c_str());
                return 1;
            }
            session_tokens.resize(n_token_count_out);
            LOG_INF("%s: loaded a session with prompt size of %d tokens\n", __func__, (int)session_tokens.size());
        }
    }

    const bool add_bos = llama_vocab_get_add_bos(vocab) && !params.use_jinja;
    if (!llama_model_has_encoder(model)) {
        GGML_ASSERT(!llama_vocab_get_add_eos(vocab));
    }

    LOG_DBG("n_ctx: %d, add_bos: %d\n", n_ctx, add_bos);

    std::vector<llama_token> embd_inp;

    bool waiting_for_first_input = false;
    auto chat_add_and_format = [&chat_msgs, &chat_templates](const std::string & role, const std::string & content) {
        common_chat_msg new_msg;
        new_msg.role = role;
        new_msg.content = content;
        auto formatted = common_chat_format_single(chat_templates.get(), chat_msgs, new_msg, role == "user", g_params->use_jinja);
        chat_msgs.push_back(new_msg);
        LOG_DBG("formatted: '%s'\n", formatted.c_str());
        return formatted;
    };

    std::string prompt;
    {
        if (params.conversation_mode && params.enable_chat_template) {
            if (!params.system_prompt.empty()) {
                // format the system prompt (will use template default if empty)
                chat_add_and_format("system", params.system_prompt);
            }

            if (!params.prompt.empty()) {
                // format and append the user prompt
                chat_add_and_format("user", params.prompt);
            } else {
                waiting_for_first_input = true;
            }

            if (!params.system_prompt.empty() || !params.prompt.empty()) {
                common_chat_templates_inputs inputs;
                inputs.use_jinja = g_params->use_jinja;
                inputs.messages = chat_msgs;
                inputs.add_generation_prompt = !params.prompt.empty();
                inputs.force_pure_content = params.force_pure_content_parser;

                prompt = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
            }
        } else {
            // otherwise use the prompt as is
            prompt = params.prompt;
        }

        if (params.interactive_first || !prompt.empty() || session_tokens.empty()) {
            LOG_DBG("tokenize the prompt\n");
            embd_inp = common_tokenize(ctx, prompt, true, true);
        } else {
            LOG_DBG("use session tokens\n");
            embd_inp = session_tokens;
        }

        LOG_DBG("prompt: \"%s\"\n", prompt.c_str());
        LOG_DBG("tokens: %s\n", string_from(ctx, embd_inp).c_str());
    }

    // Should not run without any tokens
    if (!waiting_for_first_input && embd_inp.empty()) {
        if (add_bos) {
            embd_inp.push_back(llama_vocab_bos(vocab));
            LOG_WRN("embd_inp was considered empty and bos was added: %s\n", string_from(ctx, embd_inp).c_str());
        } else {
            LOG_ERR("input is empty\n");
            return -1;
        }
    }

    // Tokenize negative prompt
    if ((int) embd_inp.size() > n_ctx - 4) {
        LOG_ERR("%s: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }

    bool session_do_save = false;

    {
        size_t n_match = 0;

        if (!session_tokens.empty()) {
            for (llama_token id : session_tokens) {
                if (n_match >= embd_inp.size() || id != embd_inp[n_match]) {
                    break;
                }
                n_match++;
            }
            if (params.prompt.empty() && n_match == embd_inp.size()) {
                LOG_INF("%s: using full prompt from session file\n", __func__);
            } else if (n_match >= embd_inp.size()) {
                LOG_INF("%s: session file has exact match for prompt!\n", __func__);
            } else if (n_match < (embd_inp.size() / 2)) {
                LOG_WRN("%s: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                        __func__, n_match, embd_inp.size());
            } else {
                LOG_INF("%s: session file matches %zu / %zu tokens of prompt\n",
                        __func__, n_match, embd_inp.size());
            }

            if (session_tokens.size() == n_match) {
                // [TAG_CONTEXT_STATE_LOGITS]
                // in this case, we are going to reuse the logits from the session
                // if we ever decide to remove the logits from the session, we need to handle this somehow
                // ref: https://github.com/ggml-org/llama.cpp/pull/18862#issuecomment-3756330941
            }

            // remove any "future" tokens that we might have inherited from the previous session
            if (session_tokens.size() > n_match) {
                if (!llama_memory_seq_rm(mem, -1, n_match, -1)) {   // 删除 不匹配的 token 对应的 KV 区间。 TODO：这里需要做分析
                    LOG_WRN("%s: unable to reuse common prefix (for example, when the memory is recurrent)\n", __func__);
                    llama_memory_clear(mem, true);
                    session_tokens.clear();
                    n_match = 0;
                } else {
                    session_tokens.resize(n_match);
                }
            }
        }

        session_do_save = !path_session.empty() && n_match < embd_inp.size() && !params.prompt_cache_ro;

        // session 里 不存 logits，
        // 加载后要靠再 llama_decode 一次最后一个 session token 把最后一档 logits 补全，
        // 如果 embd_inp 和 session_tokens 一样长，如果不 replay 得到最后一个 logits，则后面第一次 common_sampler_sample 可能不对。
        // 如果 embd_inp 比 session_tokens 长，则 embd_inp 剩余的tokens需要进行prefill，会将这次 replay 的结果丢掉
        
        if (!session_tokens.empty() && n_match > 0 && n_match == session_tokens.size()) {
            if (!common_replay_last_token(ctx, session_tokens.back(), n_match)) {
                return 1;
            }

            session_do_save = false;
            LOG_INF("%s: replayed last token from session\n", __func__);
        }
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size()) {
        params.n_keep = (int)embd_inp.size();
    } else {
        params.n_keep += add_bos; // always keep the BOS token
    }

    if (params.conversation_mode) {
        if (params.single_turn && !params.prompt.empty()) {
            params.interactive = false;
            params.interactive_first = false;
        } else {
            params.interactive_first = true;
        }
    }

    // enable interactive mode if interactive start is specified
    if (params.interactive_first) {
        params.interactive = true;
    }

    if (params.verbose_prompt) {
        LOG_INF("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            LOG_INF("%6d -> '%s'\n", embd_inp[i], common_token_to_piece(ctx, embd_inp[i]).c_str());
        }

        if (params.n_keep > add_bos) {
            LOG_INF("%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                LOG_CNT("%s", common_token_to_piece(ctx, embd_inp[i]).c_str());
            }
            LOG_CNT("'\n");
        }
        LOG_INF("\n");
    }

    // ctrl+C handling
    {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    // 打印，不会影响主流程
    if (params.interactive) {
        LOG_INF("%s: interactive mode on.\n", __func__);

        if (!params.antiprompt.empty()) {
            for (const auto & antiprompt : params.antiprompt) {
                LOG_INF("Reverse prompt: '%s'\n", antiprompt.c_str());
                if (params.verbose_prompt) {
                    auto tmp = common_tokenize(ctx, antiprompt, false, true);
                    for (int i = 0; i < (int) tmp.size(); i++) {
                        LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                    }
                }
            }
        }

        if (params.input_prefix_bos) {
            LOG_INF("Input prefix with BOS\n");
        }

        if (!params.input_prefix.empty()) {
            LOG_INF("Input prefix: '%s'\n", params.input_prefix.c_str());
            if (params.verbose_prompt) {
                auto tmp = common_tokenize(ctx, params.input_prefix, true, true);
                for (int i = 0; i < (int) tmp.size(); i++) {
                    LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                }
            }
        }

        if (!params.input_suffix.empty()) {
            LOG_INF("Input suffix: '%s'\n", params.input_suffix.c_str());
            if (params.verbose_prompt) {
                auto tmp = common_tokenize(ctx, params.input_suffix, false, true);
                for (int i = 0; i < (int) tmp.size(); i++) {
                    LOG_INF("%6d -> '%s'\n", tmp[i], common_token_to_piece(ctx, tmp[i]).c_str());
                }
            }
        }
    }

    LOG_INF("sampler seed: %u\n",     common_sampler_get_seed(smpl));
    LOG_INF("sampler params: \n%s\n", sparams.print().c_str());
    LOG_INF("sampler chain: %s\n",    common_sampler_print(smpl).c_str());

    LOG_INF("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);

    // 当开启了YaRN扩展时，尽量不要使用 group-attention
    // group-attention state
    // number of grouped KV tokens so far (used only if params.grp_attn_n > 1)
    int ga_i = 0;

    const int ga_n = params.grp_attn_n;
    const int ga_w = params.grp_attn_w;

    if (ga_n != 1) {
        GGML_ASSERT(ga_n > 0                    && "grp_attn_n must be positive");                     // NOLINT
        GGML_ASSERT(ga_w % ga_n == 0            && "grp_attn_w must be a multiple of grp_attn_n");     // NOLINT
      //GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of grp_attn_w");    // NOLINT
      //GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * grp_attn_n"); // NOLINT
        LOG_INF("self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d\n", n_ctx_train, ga_n, ga_w);
    }
    LOG_INF("\n");

    if (params.interactive) {
        const char * control_message;
        if (params.multiline_input) {
            control_message = " - To return control to the AI, end your input with '\\'.\n"
                              " - To return control without starting a new line, end your input with '/'.\n";
        } else {
            control_message = " - Press Return to return control to the AI.\n"
                              " - To return control without starting a new line, end your input with '/'.\n"
                              " - If you want to submit another line, end your input with '\\'.\n";
        }
        LOG_INF("== Running in interactive mode. ==\n");
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
        LOG_INF(       " - Press Ctrl+C to interject at any time.\n");
#endif
        LOG_INF(       "%s", control_message);
        if (params.conversation_mode && params.enable_chat_template && params.system_prompt.empty()) {
            LOG_INF(   " - Not using system message. To change it, set a different value via -sys PROMPT\n");
        }
        LOG_INF("\n");

        is_interacting = params.interactive_first;
    }

    bool is_antiprompt = false;
    bool input_echo    = true;
    bool display       = true;

    int n_past             = 0;
    int n_remain           = params.n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;

    std::vector<int>   input_tokens;  g_input_tokens  = &input_tokens;
    std::vector<int>   output_tokens; g_output_tokens = &output_tokens;
    std::ostringstream output_ss;     g_output_ss     = &output_ss;
    std::ostringstream assistant_ss; // for storing current assistant message, used in conversation mode

    // the first thing we will do is to output the prompt, so set color accordingly
    console::set_display(DISPLAY_TYPE_PROMPT);
    display = params.display_prompt;

    std::vector<llama_token> embd;

    // single-token antiprompts
    std::vector<llama_token> antiprompt_token;

    for (const std::string & antiprompt : params.antiprompt) {
        auto ids = ::common_tokenize(ctx, antiprompt, false, true);
        if (ids.size() == 1) {
            antiprompt_token.push_back(ids[0]);
        }
    }

    if (llama_model_has_encoder(model)) {
        int enc_input_size = embd_inp.size();
        llama_token * enc_input_buf = embd_inp.data();

        if (llama_encode(ctx, llama_batch_get_one(enc_input_buf, enc_input_size))) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return 1;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }

        embd_inp.clear();
        embd_inp.push_back(decoder_start_token_id);
    }

    // n_remain ： decode 过程中还剩多少token没有生成
    // is_antiprompt : 是否是反向提示词
    // params.interactive : 是否是交互模式
    while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
        // predict
        if (!embd.empty()) {
            // Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            // 4 :  是沿用了多年的 小余量（slack），用来和 prompt 最长允许 token 数 对齐，
            // 并给下面这些留个缓冲，避免 贴满 n_ctx 边界 时和别处逻辑打架
            int max_embd_size = n_ctx - 4;

            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int) embd.size() > max_embd_size) {
                const int skipped_tokens = (int) embd.size() - max_embd_size;
                embd.resize(max_embd_size);

                console::set_display(DISPLAY_TYPE_ERROR);
                LOG_WRN("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
                console::set_display(DISPLAY_TYPE_RESET);
            }

            //  group-attention 或者 Self-Extend 的原理：
            // 模型在 训练长度 n_ctx_train 内，RoPE 位置编码、注意力对“位置”的分布是给定的；
            //      序列一味变长时，绝对位置标号会超出训练时见过的范围，质量容易掉。
            // Self-Extend 的做法不是无限加长度而不处理，而是：在一段已经写进 KV 的区间里，把 多个相邻 token 在逻辑上看成“更粗一格”的位置——等价于对 KV 里保存的位置下标做分组/缩放，让当前 query 看到的 有效位置范围仍在模型更熟悉的尺度里，从而 用固定物理 n_ctx 承载更长的逻辑序列（近似“远处历史变粗、近处仍细”）。

            // ga_n（grp_attn_n）
            //    分组因子：在某个窗口里对 KV 做 llama_memory_seq_div(..., ga_n)，
            //             把该区间内每个 cell 的 逻辑位置除以 ga_n（实现里还对 shift 做了累计，便于 RoPE 一致）。
            //    每 ga_n 个历史位置在位置编码上“合并成一格”，从而 拉长等效能看到的上下文（代价是对那一段历史的区分变粗）。

            // ga_w（grp_attn_w）
            //     窗口宽度：代码要求 ga_w % ga_n == 0。当 n_past >= ga_i + ga_w 时进入维护循环：
            //              在从 ga_i 起的一层滑动策略下，对长度为 ga_w 的那一段 KV 区间做 seq_div（除以 ga_n），
            //              再配合两次 llama_memory_seq_add 调整前段/后段位置，并减少 n_past（压缩已占用“位置计数”）。
            // 也就是说：每积累够一段宽度为 ga_w 的序列，就对其中一块做对齐到 ga_n 的压缩，避免一直线性涨满 n_ctx。

            // 如果已用 YaRN 把 n_ctx 配满需求，通常 grp_attn_n 保持默认 1 即可
            if (ga_n == 1) {
                // ga_n == 1：不启用 Self-Extend；
                // 吃满 context 走的是 context shift（扔掉一半非保留区、seq_rm/seq_add），和 group-attention 无关。

                // infinite text generation via context shifting
                // if we run out of context:
                // - take the n_keep first tokens from the original prompt (via n_past)
                // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches

                // 已有 KV 占用了 n_past 个位置，本批还要再塞进 embd.size() 个 token，加起来 ≥ 上下文长度 n_ctx，
                // 再不加处理就会溢出或无法写入。
                // n_past：已经写进 KV、逻辑上已处理的 token 数。
                // embd：当前这一批准备再喂进去的 token。
                if (n_past + (int) embd.size() >= n_ctx) {
                    // 不允许滑动：直接告警并 break 退出主循环，生成停止。
                    if (!params.ctx_shift){
                        LOG_WRN("\n\n%s: context full and context shift is disabled => stopping\n", __func__);
                        break;
                    }

                    // 特殊模式（注释里是“context full 时停”这一类约定）：同样 break。
                    if (params.n_predict == -2) {
                        LOG_WRN("\n\n%s: context full and n_predict == %d => stopping\n", __func__, params.n_predict);
                        break;
                    }

                    // 在可丢区域里 扔掉一半（不是全丢，也不是只丢 1），给后面 腾出约 n_discard 个槽位。
                    const int n_left    = n_past - params.n_keep;
                    const int n_discard = n_left/2;

                    LOG_DBG("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                            n_past, n_left, n_ctx, params.n_keep, n_discard);

                    // 对序列 0，删掉 逻辑位置 在 [n_keep, n_keep + n_discard) 那一段的 KV 槽位/内容（中间被挖掉一块）。
                    llama_memory_seq_rm (mem, 0, params.n_keep            , params.n_keep + n_discard);
                    // 把 [n_keep + n_discard, n_past) 里还剩下的那些格子的 位置编号整体减 n_discard，
                    // 让序列在位置轴上 重新接成连续，缺口被“填上”。
                    llama_memory_seq_add(mem, 0, params.n_keep + n_discard, n_past, -n_discard);

                    n_past -= n_discard;

                    LOG_DBG("after swap: n_past = %d\n", n_past);

                    LOG_DBG("embd: %s\n", string_from(ctx, embd).c_str());

                    LOG_DBG("clear session path\n");
                    // Session 文件里存的是 与当时 KV 布局一致的快照。做完 shift 后 KV 与 token 序列的对应关系已变，旧 session 不再可信，
                    // 所以不要继续用这个路径自动存/恢复，清空 session 路径。
                    path_session.clear();
                }
            } else {
                // ga_n > 1：走 Self-Extend 分支，用 seq_div + seq_add 维护 KV 位置，而不是简单删一半。
                // context extension via Self-Extend
                while (n_past >= ga_i + ga_w) {
                    const int ib = (ga_n*ga_i)/ga_w;
                    const int bd = (ga_w/ga_n)*(ga_n - 1);
                    const int dd = (ga_w/ga_n) - ib*bd - ga_w;

                    LOG_DBG("\n");
                    LOG_DBG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i, n_past, ib*bd, ga_i + ib*bd, n_past + ib*bd);
                    LOG_DBG("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", ga_i + ib*bd, ga_i + ib*bd + ga_w, ga_n, (ga_i + ib*bd)/ga_n, (ga_i + ib*bd + ga_w)/ga_n);
                    LOG_DBG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i + ib*bd + ga_w, n_past + ib*bd, dd, ga_i + ib*bd + ga_w + dd, n_past + ib*bd + dd);

                    llama_memory_seq_add(mem, 0, ga_i,                n_past,              ib*bd);
                    llama_memory_seq_div(mem, 0, ga_i + ib*bd,        ga_i + ib*bd + ga_w, ga_n);
                    llama_memory_seq_add(mem, 0, ga_i + ib*bd + ga_w, n_past + ib*bd,      dd);

                    n_past -= bd;

                    ga_i += ga_w/ga_n;

                    LOG_DBG("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", n_past + bd, n_past, ga_i);
                }
            }

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            // prefix caching：尝试从加载的会话中重用匹配的前缀，而不是重新评估（通过 n_past）。
            if (n_session_consumed < (int) session_tokens.size()) {
                size_t i = 0;
                for ( ; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int) session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            if (!embd.empty()) {
                const bool is_last_batch = (n_consumed >= (int) embd_inp.size());
                const bool save_now = session_do_save && is_last_batch;
                // 在prefill阶段，对很多 n_batch 子批进行处理的过程中，KV cache已经生成了
                // 在对最后一个子批的 n_eval( <= n_batch)个tokens处理时，会将前 n_eval - 1 个token的 KV cache以及之前子批的KV cache写入文件
                //                      最后一个token的KV cache并不写入文件
                // 原因解释： recurrent/hybrid模型的KV cache不支持移除token
                // 规避方法： 
                //    1. 将所有的token，包含上述的最后一个token，都存入session，但不存最后一个token的 KV cache
                //    2. 在从session中读的数据，如果和当前输入，刚好匹配到上述的最后一个token，
                //       那么对这个token做一次replay，得到它的 KV，并算出logits，这样在sampler处理这个logits时不会出错
                if (!common_prompt_batch_decode(ctx, embd, n_past, params.n_batch, path_session, save_now)) {
                    return 1;
                }
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = session_tokens.size();
                session_do_save = false;

                LOG_DBG("n_past = %d\n", n_past);

                // Display total tokens alongside total time
                if (params.n_print > 0 && n_past % params.n_print == 0) {
                    LOG_DBG("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", n_past, n_ctx);
                }
            }
        }

        embd.clear();

        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
            // 准备 decode 的输入

            const llama_token id = common_sampler_sample(smpl, ctx, -1);

            common_sampler_accept(smpl, id, /* accept_grammar= */ true);

            // LOG_DBG("last: %s\n", string_from(ctx, smpl->prev.to_vector()).c_str());

            embd.push_back(id);

            if (params.conversation_mode && !waiting_for_first_input && !llama_vocab_is_eog(vocab, id)) {
                assistant_ss << common_token_to_piece(ctx, id, false);
            }

            // echo this to console
            input_echo = true;

            // decrement remaining sampling budget
            --n_remain;

            LOG_DBG("n_remain: %d\n", n_remain);
        } else {
            // 准备 prefill 阶段的 u_batch
            // some user input remains from prompt or interaction, forward it to processing
            LOG_DBG("embd_inp.size(): %d, n_consumed: %d\n", (int) embd_inp.size(), n_consumed);
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                common_sampler_accept(smpl, embd_inp[n_consumed], /* accept_grammar= */ false);

                ++n_consumed;

                // 一次加入 n_batch 个 tokens 进行 prefill，embd中的 token 数有可能小于 n_batch
                if ((int) embd.size() == params.n_batch) {
                    break;
                }
            }
        }

        // display text
        if (input_echo && display) {
            for (auto id : embd) {
                const std::string token_str = common_token_to_piece(ctx, id, params.special);

                // Console/Stream Output
                LOG("%s", token_str.c_str());

                // Record Displayed Tokens To Log
                // Note: Generated tokens are created one by one hence this check
                if (embd.size() > 1) {
                    // Incoming Requested Tokens
                    input_tokens.push_back(id);
                } else {
                    // Outgoing Generated Tokens
                    output_tokens.push_back(id);
                    output_ss << token_str;
                }
            }
        }

        // reset color to default if there is no pending user input
        if (input_echo && (int) embd_inp.size() == n_consumed) {
            console::set_display(DISPLAY_TYPE_RESET);
            display = true;
        }

        // if not currently processing queued inputs;
        // 当前输入的 tokens 已经被取完了
        if ((int) embd_inp.size() <= n_consumed) {
            // check for reverse prompt in the last n_prev tokens
            if (!params.antiprompt.empty()) {
                const int n_prev = 32;

                // 给上层做「看最近 模型/用户 已经接受了哪些 token，对应的明文长什么样」
                // 例如在 completion 里查 antiprompt 是否出现在输出尾部，或类似需要「短窗明文」的地方。
                const std::string last_output = common_sampler_prev_str(smpl, ctx, n_prev);

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                // If we're not running interactively, the reverse prompt might be tokenized with some following characters
                // so we'll compensate for that by widening the search window a bit.

                // 在最后32个token对应的字符串中，查找反向提示词是否出现在输出尾部
                for (std::string & antiprompt : params.antiprompt) {
                    size_t extra_padding = params.interactive ? 0 : 2;
                    size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                        ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                        : 0;

                    if (last_output.find(antiprompt, search_start_pos) != std::string::npos) {
                        if (params.interactive) {
                            is_interacting = true;
                        }
                        is_antiprompt = true;
                        break;
                    }
                }

                // check for reverse prompt using special tokens
                // avoid calling common_sampler_last() if last_output is empty
                if (!last_output.empty()) {
                    llama_token last_token = common_sampler_last(smpl);
                    for (auto token : antiprompt_token) {
                        if (token == last_token) {
                            if (params.interactive) {
                                is_interacting = true;
                            }
                            is_antiprompt = true;
                            break;
                        }
                    }
                }

                if (is_antiprompt) {
                    LOG_DBG("found antiprompt: %s\n", last_output.c_str());
                }
            }

            // deal with end of generation tokens in interactive mode
            if (!waiting_for_first_input && llama_vocab_is_eog(vocab, common_sampler_last(smpl))) {
                LOG_DBG("found an EOG token\n");

                if (params.interactive) {
                    if (!params.antiprompt.empty()) {
                        // 交互模式下，当最后一个token是 EOG 时，将反向提示词转换为 token 并注入到 embd_inp 中，等待用户输入
                        // tokenize and inject first reverse prompt
                        const auto first_antiprompt = common_tokenize(ctx, params.antiprompt.front(), false, true);
                        embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                        is_antiprompt = true;
                    }

                    if (params.enable_chat_template) {
                        chat_add_and_format("assistant", assistant_ss.str());
                    }
                    is_interacting = true;
                    LOG("\n");
                }
            }

            if (params.conversation_mode && !waiting_for_first_input) {
                if (!prompt.empty()) {
                    prompt.clear();
                    is_interacting = false;
                }
            }

            if ((n_past > 0 || waiting_for_first_input) && is_interacting) {
                LOG_DBG("waiting for user input\n");

                if (params.conversation_mode) {
                    LOG("\n> ");    // 会话模式打 \n> ，表示可以输入。
                }

                if (params.input_prefix_bos) {
                    LOG_DBG("adding input prefix BOS token\n");
                    embd_inp.push_back(llama_vocab_bos(vocab));
                }

                std::string buffer;
                if (!params.input_prefix.empty() && !params.conversation_mode) {
                    LOG_DBG("appending input prefix: '%s'\n", params.input_prefix.c_str());
                    LOG("%s", params.input_prefix.c_str());
                }

                // color user input only
                console::set_display(DISPLAY_TYPE_USER_INPUT);
                display = params.display_prompt;

                // 在 is_interacting、且前面 embd_inp 已消费完
                // 场景是 交互 / 会话：第一轮可能用了 params.prompt 拼好的首屏；之后每一轮用户接着打字，就用 buffer 收集，
                // 再 tokenize 成 token 追加到 embd_inp，继续 decode，而不是再从 params.prompt 读一遍。
                std::string line;
                bool another_line = true;
                do {
                    // readline 读的是 当前在终端里用户新敲的一轮输入
                    another_line = console::readline(line, params.multiline_input);
                    buffer += line;
                } while (another_line);

                // done taking input, reset color
                console::set_display(DISPLAY_TYPE_RESET);
                display = true;

                if (buffer.empty()) { // Ctrl+D on empty line exits
                    LOG("EOF by user\n");
                    break;
                }

                if (buffer.back() == '\n') {
                    // Implement #587:
                    // If the user wants the text to end in a newline,
                    // this should be accomplished by explicitly adding a newline by using \ followed by return,
                    // then returning control by pressing return again.
                    buffer.pop_back();
                }

                if (buffer.empty()) { // Enter key on empty line lets the user pass control back
                    LOG_DBG("empty line, passing control back\n");
                } else { // Add tokens to embd only if the input buffer is non-empty
                    // append input suffix if any
                    if (!params.input_suffix.empty() && !params.conversation_mode) {
                        LOG_DBG("appending input suffix: '%s'\n", params.input_suffix.c_str());
                        LOG("%s", params.input_suffix.c_str());
                    }

                    LOG_DBG("buffer: '%s'\n", buffer.c_str());

                    const size_t original_size = embd_inp.size();

                    if (params.escape) {
                        string_process_escapes(buffer);
                    }

                    bool format_chat = params.conversation_mode && params.enable_chat_template;
                    std::string user_inp = format_chat
                        ? chat_add_and_format("user", std::move(buffer))
                        : std::move(buffer);
                    // TODO: one inconvenient of current chat template implementation is that we can't distinguish between user input and special tokens (prefix/postfix)
                    const auto line_pfx = common_tokenize(ctx, params.input_prefix, false, true);
                    const auto line_inp = common_tokenize(ctx, user_inp,            false, format_chat);
                    const auto line_sfx = common_tokenize(ctx, params.input_suffix, false, true);

                    LOG_DBG("input tokens: %s\n", string_from(ctx, line_inp).c_str());

                    // if user stop generation mid-way, we must add EOT to finish model's last response
                    if (need_insert_eot && format_chat) {
                        llama_token eot = llama_vocab_eot(vocab);
                        embd_inp.push_back(eot == LLAMA_TOKEN_NULL ? llama_vocab_eos(vocab) : eot);
                        need_insert_eot = false;
                    }

                    embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
                    embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());

                    if (params.verbose_prompt) {
                        LOG_INF("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size() - original_size);
                    }

                    for (size_t i = original_size; i < embd_inp.size(); ++i) {
                        const llama_token token = embd_inp[i];
                        const std::string token_str = common_token_to_piece(ctx, token);
                        output_tokens.push_back(token);
                        output_ss << token_str;

                        if (params.verbose_prompt) {
                            LOG_INF("%6d -> '%s'\n", token, token_str.c_str());
                        }
                    }

                    // reset assistant message
                    assistant_ss.str("");

                    n_remain -= line_inp.size();
                    LOG_DBG("n_remain: %d\n", n_remain);
                }

                input_echo = false; // do not echo this again
            }

            if (n_past > 0 || waiting_for_first_input) {
                if (is_interacting) {
                    common_sampler_reset(smpl);
                }
                is_interacting = false;

                if (waiting_for_first_input && params.single_turn) {
                    params.interactive = false;
                    params.interactive_first = false;
                }
                waiting_for_first_input = false;
            }
        }

        // end of generation
        // 在非交互模型下，当前输入的最后一个 token 是 EOG token，则结束生成
        if (!embd.empty() && llama_vocab_is_eog(vocab, embd.back()) && !(params.interactive)) {
            LOG(" [end of text]\n");
            break;
        }

        // 在交互模式下，尊重最大 token 数限制，达到后回退到用户输入
        // 当 n_predict == -1 （无限）或 -2 （达到上下文大小）时跳过此逻辑
        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        // We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
        if (params.interactive && n_remain <= 0 && params.n_predict >= 0) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }

    // 保存 session
    // 要求：params.prompt_cache_all 为 true ： 配置要求将用户输入和生成结果都保存到 session 中
    //      params.prompt_cache_ro 为 false ： 配置要求 session 文件是可写的
    
    // 仅就 save_now 那次保存而言：不会把前面各子批的 token id 写进 session 里那段 token 记录，只有最后子批（且是「减 1」那条路径下的那一段）会写进头里。
    // 若开了 prompt_cache_all 且程序正常结束：前面所有子批的 id 会通过最后的 session_tokens 整体落盘。
    // 若没开 prompt_cache_all：前面子批的 id 一般不会以完整列表形式出现在 session 文件里，只有最后一次子批相关的一小段会出现在头里（若走了那段保存逻辑）。
    if (!path_session.empty() && params.prompt_cache_all && !params.prompt_cache_ro) {
        LOG("\n%s: saving final output to session file '%s'\n", __func__, path_session.c_str());
        llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
    }

    LOG("\n\n");
    common_perf_print(ctx, smpl);

    llama_backend_free();

    ggml_threadpool_free_fn(threadpool);
    ggml_threadpool_free_fn(threadpool_batch);

    return 0;
}
