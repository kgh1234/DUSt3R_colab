#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
import torch
import tempfile
import time  # ⏳ 실행 시간 측정을 위한 라이브러리 추가

from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.demo import get_args_parser, main_demo, set_print_with_timestamp

import matplotlib.pyplot as pl
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # ✅ Ampere 이후 GPU 최적화

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    set_print_with_timestamp()

    print("🚀 DUSt3R Gradio Demo 시작...")  # 실행 시작 로그

    # 📂 임시 디렉토리 설정
    if args.tmp_dir is not None:
        tmp_path = args.tmp_dir
        os.makedirs(tmp_path, exist_ok=True)
        tempfile.tempdir = tmp_path

    # 🌍 서버 주소 설정 (외부 접속 허용)
    server_name = args.server_name if args.server_name is not None else ('0.0.0.0' if args.local_network else '127.0.0.1')

    # 🔗 모델 가중치 경로 설정
    weights_path = args.weights if args.weights is not None else "naver/" + args.model_name

    # 📥 모델 로딩
    print("📥 모델 불러오는 중...")
    start_time = time.time()
    model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(args.device)
    print(f"✅ 모델 로드 완료! (소요 시간: {time.time() - start_time:.2f}초)")

    # 📂 Gradio UI 실행을 위한 임시 디렉토리 생성
    with tempfile.TemporaryDirectory(suffix='dust3r_gradio_demo') as tmpdirname:
        print(f"📁 출력 디렉토리: {tmpdirname}")

        print("🚀 Gradio 인터페이스 실행 중...")
        start_time = time.time()

        # ✅ Gradio 실행 (공유 링크 포함)
        main_demo(
            tmpdirname, 
            model, 
            args.device, 
            args.image_size, 
            "0.0.0.0",  # ✅ 외부 접속 허용 (기본값: 127.0.0.1)
            args.server_port, 
            silent=args.silent,
            
        )

        # ✅ 실행 완료 메시지
        print(f"✅ Gradio 실행 완료! (소요 시간: {time.time() - start_time:.2f}초)")
        print("🔗 UI가 준비되었습니다! 아래 링크를 클릭하세요:")
        print(f"🌍 Local URL: http://127.0.0.1:{args.server_port} (로컬 접속)")
        print(f"🌍 Public Gradio Link: {server_name}")  # ✅ 공유된 링크 표시
