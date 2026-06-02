document.getElementById('downloadImage').addEventListener('click', function() {
    const url = "http://localhost:8080/download/"; // FastAPI 서버의 다운로드 엔드포인트

    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error('파일 다운로드 실패: ' + response.status);
            }
            return response.blob(); // Blob 형태로 변환
        })
        .then(blob => {
            const url = window.URL.createObjectURL(blob); // Blob URL 생성
            const a = document.createElement('a'); // <a> 태그 생성
            a.style.display = 'none'; // 보이지 않도록 설정
            a.href = url;
            a.download = 'stock.png'; // 다운로드할 파일 이름
            document.body.appendChild(a); // DOM에 추가
            a.click(); // 클릭 이벤트 발생
            window.URL.revokeObjectURL(url); // 생성한 URL 해제
            console.log("파일이 성공적으로 다운로드되었습니다.");
        })
        .catch(error => {
            console.error(error);
        });
});
