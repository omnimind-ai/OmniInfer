/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every LICENSELINT

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pytorch/tokenizers/error.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>
#include <pytorch/tokenizers/result.h>
#include <pytorch/tokenizers/sentencepiece.h>
#include <pytorch/tokenizers/tekken.h>
#include <pytorch/tokenizers/tiktoken.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace py = pybind11;
using namespace tokenizers;

// Helper function to convert Result<T> to Python
template <typename T>
T unwrap_result(const Result<T>& result) {
  if (result.error() != Error::Ok) {
    std::string error_msg;
    switch (result.error()) {
      case Error::Ok:
        error_msg = "Ok";
        break;
      case Error::Internal:
        error_msg = "Internal";
        break;
      case Error::Uninitialized:
        error_msg = "Uninitialized";
        break;
      case Error::OutOfRange:
        error_msg = "OutOfRange";
        break;
      case Error::LoadFailure:
        error_msg = "LoadFailure";
        break;
      case Error::EncodeFailure:
        error_msg = "EncodeFailure";
        break;
      case Error::Base64DecodeFailure:
        error_msg = "Base64DecodeFailure";
        break;
      case Error::ParseFailure:
        error_msg = "ParseFailure";
        break;
      case Error::DecodeFailure:
        error_msg = "DecodeFailure";
        break;
      case Error::RegexFailure:
        error_msg = "RegexFailure";
        break;
      default:
        error_msg = "Unknown error";
        break;
    }
    throw std::runtime_error("Tokenizer error: " + error_msg);
  }
  return result.get();
}

PYBIND11_MODULE(pytorch_tokenizers_cpp, m) {
  m.doc() = "PyTorch Tokenizers Python bindings";

  // Bind Error enum
  py::enum_<Error>(m, "Error")
      .value("Ok", Error::Ok)
      .value("Internal", Error::Internal)
      .value("Uninitialized", Error::Uninitialized)
      .value("OutOfRange", Error::OutOfRange)
      .value("LoadFailure", Error::LoadFailure)
      .value("EncodeFailure", Error::EncodeFailure)
      .value("Base64DecodeFailure", Error::Base64DecodeFailure)
      .value("ParseFailure", Error::ParseFailure)
      .value("DecodeFailure", Error::DecodeFailure)
      .value("RegexFailure", Error::RegexFailure);

  // Bind TokenIndex struct
  py::class_<TokenIndex>(m, "TokenIndex")
      .def_readonly("str", &TokenIndex::str)
      .def_readonly("id", &TokenIndex::id);

  // Bind base Tokenizer class
  py::class_<Tokenizer>(m, "Tokenizer")
      .def(
          "load",
          [](Tokenizer& self, const std::string& tokenizer_path) {
            Error error = self.load(tokenizer_path);
            if (error != Error::Ok) {
              throw std::runtime_error("Failed to load tokenizer");
            }
          },
          py::arg("tokenizer_path"))
      .def(
          "encode",
          [](const Tokenizer& self,
             const std::string& input,
             int8_t bos,
             int8_t eos) {
            return unwrap_result(self.encode(input, bos, eos));
          },
          py::arg("input"),
          py::arg("bos") = 0,
          py::arg("eos") = 0)
      .def(
          "decode",
          [](const Tokenizer& self,
             uint64_t token,
             bool skip_special_tokens = false) {
            return unwrap_result(
                self.decode(token, token, skip_special_tokens));
          },
          py::arg("token"),
          py::arg("skip_special_tokens") = false)
      .def("vocab_size", &Tokenizer::vocab_size)
      .def("bos_tok", &Tokenizer::bos_tok)
      .def("eos_tok", &Tokenizer::eos_tok)
      .def("is_loaded", &Tokenizer::is_loaded);

  // Bind HFTokenizer
  py::class_<HFTokenizer, Tokenizer>(m, "HFTokenizer")
      .def(py::init<>())
      .def(
          "load",
          [](HFTokenizer& self, const std::string& tokenizer_path) {
            Error error = self.load(tokenizer_path);
            if (error != Error::Ok) {
              throw std::runtime_error("Failed to load HF tokenizer");
            }
          },
          py::arg("tokenizer_path"))
      .def(
          "encode",
          [](const HFTokenizer& self,
             const std::string& input,
             int8_t bos,
             int8_t eos) {
            return unwrap_result(self.encode(input, bos, eos));
          },
          py::arg("input"),
          py::arg("bos") = 0,
          py::arg("eos") = 0)
      .def(
          "decode",
          [](const HFTokenizer& self,
             uint64_t token,
             bool skip_special_tokens = false) {
            return unwrap_result(
                self.decode(token, token, skip_special_tokens));
          },
          py::arg("token"),
          py::arg("skip_special_tokens") = false)
      .def(
          "decode_batch",
          [](const HFTokenizer& self,
             const std::vector<uint64_t>& tokens,
             bool skip_special_tokens = false) {
            return unwrap_result(self.decode(tokens, skip_special_tokens));
          },
          py::arg("tokens"),
          py::arg("skip_special_tokens") = false);

  // Bind Tiktoken
  py::class_<Tiktoken, Tokenizer>(m, "Tiktoken")
      .def(py::init<>())
      .def(
          py::init<std::vector<std::string>, size_t, size_t>(),
          py::arg("special_tokens"),
          py::arg("bos_token_index"),
          py::arg("eos_token_index"))
      .def(
          py::init<std::string, std::vector<std::string>, size_t, size_t>(),
          py::arg("pattern"),
          py::arg("special_tokens"),
          py::arg("bos_token_index"),
          py::arg("eos_token_index"))
      .def(
          "load",
          [](Tiktoken& self, const std::string& tokenizer_path) {
            Error error = self.load(tokenizer_path);
            if (error != Error::Ok) {
              throw std::runtime_error("Failed to load Tiktoken tokenizer");
            }
          },
          py::arg("tokenizer_path"))
      .def(
          "encode",
          [](const Tiktoken& self,
             const std::string& input,
             int8_t bos,
             int8_t eos) {
            return unwrap_result(self.encode(input, bos, eos));
          },
          py::arg("input"),
          py::arg("bos") = 0,
          py::arg("eos") = 0)
      .def(
          "decode",
          [](const Tiktoken& self,
             uint64_t token,
             bool skip_special_tokens = false) {
            return unwrap_result(
                self.decode(token, token, skip_special_tokens));
          },
          py::arg("token"),
          py::arg("skip_special_tokens") = false);

  // Bind Llama2cTokenizer
  py::class_<Llama2cTokenizer, Tokenizer>(m, "Llama2cTokenizer")
      .def(py::init<>())
      .def(
          "load",
          [](Llama2cTokenizer& self, const std::string& tokenizer_path) {
            Error error = self.load(tokenizer_path);
            if (error != Error::Ok) {
              throw std::runtime_error("Failed to load Llama2c tokenizer");
            }
          },
          py::arg("tokenizer_path"))
      .def(
          "encode",
          [](const Llama2cTokenizer& self,
             const std::string& input,
             int8_t bos,
             int8_t eos) {
            return unwrap_result(self.encode(input, bos, eos));
          },
          py::arg("input"),
          py::arg("bos") = 0,
          py::arg("eos") = 0)
      .def(
          "decode",
          [](const Llama2cTokenizer& self,
             uint64_t token,
             bool skip_special_tokens = false) {
            return unwrap_result(
                self.decode(token, token, skip_special_tokens));
          },
          py::arg("token"),
          py::arg("skip_special_tokens") = false);

  // Bind SPTokenizer (SentencePiece)
  py::class_<SPTokenizer, Tokenizer>(m, "SPTokenizer")
      .def(py::init<>())
      .def(
          "load",
          [](SPTokenizer& self, const std::string& tokenizer_path) {
            Error error = self.load(tokenizer_path);
            if (error != Error::Ok) {
              throw std::runtime_error(
                  "Failed to load SentencePiece tokenizer");
            }
          },
          py::arg("tokenizer_path"))
      .def(
          "encode",
          [](const SPTokenizer& self,
             const std::string& input,
             int8_t bos,
             int8_t eos) {
            return unwrap_result(self.encode(input, bos, eos));
          },
          py::arg("input"),
          py::arg("bos") = 0,
          py::arg("eos") = 0)
      .def(
          "decode",
          [](const SPTokenizer& self,
             uint64_t token,
             bool skip_special_tokens = false) {
            return unwrap_result(
                self.decode(token, token, skip_special_tokens));
          },
          py::arg("token"),
          py::arg("skip_special_tokens") = false);

  // Bind Tekken tokenizer
  py::class_<Tekken, Tokenizer>(m, "Tekken")
      .def(py::init<>())
      .def(
          "load",
          [](Tekken& self, const std::string& tokenizer_path) {
            Error error = self.load(tokenizer_path);
            if (error != Error::Ok) {
              throw std::runtime_error("Failed to load Tekken tokenizer");
            }
          },
          py::arg("tokenizer_path"))
      .def(
          "encode",
          [](const Tekken& self,
             const std::string& input,
             int8_t bos,
             int8_t eos) {
            return unwrap_result(self.encode(input, bos, eos));
          },
          py::arg("input"),
          py::arg("bos") = 0,
          py::arg("eos") = 0)
      .def(
          "decode",
          [](const Tekken& self,
             uint64_t token,
             bool skip_special_tokens = false) {
            return unwrap_result(
                self.decode(token, token, skip_special_tokens));
          },
          py::arg("token"),
          py::arg("skip_special_tokens") = false)
      .def(
          "decode_batch",
          [](const Tekken& self,
             const std::vector<uint64_t>& tokens,
             bool skip_special_tokens = false) {
            std::string result;
            for (size_t i = 0; i < tokens.size(); ++i) {
              uint64_t prev_token = (i == 0) ? 0 : tokens[i - 1];
              auto decoded =
                  self.decode(prev_token, tokens[i], skip_special_tokens);
              if (decoded.error() != Error::Ok) {
                throw std::runtime_error("Failed to decode token");
              }
              result += decoded.get();
            }
            return result;
          },
          py::arg("tokens"),
          py::arg("skip_special_tokens") = false)
      .def("vocab_size", &Tekken::vocab_size)
      .def("bos_tok", &Tekken::bos_tok)
      .def("eos_tok", &Tekken::eos_tok)
      .def("is_loaded", &Tekken::is_loaded)
      .def("get_version", &Tekken::get_version);
}
