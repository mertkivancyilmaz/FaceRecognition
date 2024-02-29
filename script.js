//Sisteme HTML dosyasında oluşturduğumuz İnput ile resim yüklemesi yapıyoruz
const imageUpload = document.getElementById('imageUpload')
//Alttaki bütün modelleri aynı anda başlatabilmek için (Promise.all)kullanıyoruz
Promise.all([
//İndirdiğimiz hazır kütüphane olan faceapi'yi yüzleri tanımak için kullanıyoruz
// ve yol olarak /modelse giden yolu gösteriyoruz. 3 Adet indirilen modelleri kullanıyoruz
  faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
]).then(start)
//Yüklemeler bittiğinde then.(start)ile başlatıyoruz

async function start() {
  //Basit bir div oluşturuyoruz
  const container = document.createElement('div')
  //Ayarlamak istediğimiz pozisyonu belirtiyoruz
  container.style.position = 'relative'
  //Container diye bir değer tanımlıyoruz ve bunu aşağıda tekrar tekrar kullanıyoruz
  document.body.append(container)

  const labeledFaceDescriptors = await loadLabeledImages()
  //Yüz eşleştirmesi için FaceMatcher adında değer tanımlıyoruz
  //Algoritmanın doğruluk toleransını 0.6 olarak standart değer belirliyoruz
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6)
  //Resmi Getir
  let image
  //Yüz apisi ile çıkan canvas'ı getir
  let canvas
  //Loaded Yazısı
  document.body.append('Loaded')
  //Görüntü yüklediğimize kodu buraya çağırıyoruz, apiyi çağırınca async kod olacağından asycn kullanıyoruz
  imageUpload.addEventListener('change', async () => {
    //Resmin dışında yanlış yerleri kapsamaması if (image)kullanıyoruz 
    if (image) image.remove()
    //Aynı şekilde yanlış eşleşme olmaması için if(canvas) kullanıyoruz.
    if (canvas) canvas.remove()
    //Resim faceapi'mize eşit olacak ve resim yüklememizi yapacağız, aynı zamanda eşzamansız olduğundan "await kullanıyoruz"
    image = await faceapi.bufferToImage(imageUpload.files[0])
    //Burada seçilen resmi ekrana getiriyoruz
    container.append(image)
    //Yüz apisine eşit yeni bir canvar yaratıyor ve yeni tuval yaratıyor
    canvas = faceapi.createCanvasFromMedia(image)
    //Yüz apisine eşit canvas oluşturuyor
    container.append(canvas)
    //Görüntüyü yeniden boyutlandırıyoruz
    const displaySize = { width: image.width, height: image.height }
    faceapi.matchDimensions(canvas, displaySize)
    //Yüzlerde tespit etmek istediğimiz alanları bekletiyoruz, yüklediğimiz görüntüden sonra yapmak istiyoruz
    //WirhFaceLandMarks farklı yüzlerin yerini tespit etmede kullanılır
    //withFaceDescriptors yüzlerin etrafında çizgiler çekerek yüzleri gösterir
    const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors()
    //Yeniden tespit edilen yüzleri boyutlandıtıyoruz
    const resizedDetections = faceapi.resizeResults(detections, displaySize)
    //Yüklediğimiz tüm görüntüleri inceleyerek doğruluğu arttıracaktır
    const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))
    //Tespitleri her bir yüz için uygulama
    results.forEach((result, i) => {
      //Birden fazla fotoğraf olduğundan döngü şeklinde yapoıyoruz ve doğruluk oranını arttırıyoruz
      const box = resizedDetections[i].detection.box
      //Nokta algılama methodu ile yüzü tespit ediyoruz
      //Yüzlerin algılanması için hepsine isim etiketi koyuyoruz
      const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() })
      //Yüz için kutu canvas çiziliyor
      drawBox.draw(canvas)
    })
  })
}

//Yeni fonksiyon tanımlıyoruz. Bu fonksiyon yüklenen resimlere Label tanımlar
function loadLabeledImages() {
  //Siteden almak istediğimiz dosyaları belirtiyoruz
  const labels = ['Black Widow', 'Captain America', 'Captain Marvel', 'Hawkeye', 'Jim Rhodes', 'Thor', 'Tony Stark']
  //Hespi aynı anda çalışması için "promise" kullanılıyor
  return Promise.all(
    //Tüm etiketler buradaki dizide dönmesi için dögü başlatıyrouz
    labels.map(async label => {
      const descriptions = []
      //Her fotoğraf için 2 resim koyduk, 2 resim yeterli hata payını azaltmamız için uygun. Her kişide 2 adet resim olduğundan FOR döngüsü 0-2 arasında olacak
      for (let i = 1; i <= 2; i++) {
        //resimlerin alınacağı yolu seçiyoruz
        const img = await faceapi.fetchImage(`https://raw.githubusercontent.com/WebDevSimplified/Face-Recognition-JavaScript/master/labeled_images/${label}/${i}.jpg`)
        //Face Api tek bir yüzü algılacayacak diyoruz ve bunu landmarks ile yapmak istiyoruz
        const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
        //Burada algılanan yüzü tarif eden descriptor metodunu kullanıyoruz ve gelen yüzün kime ait olduğunnu görüyoruz
        descriptions.push(detections.descriptor)
      }
      //Bu döngüden çıkmak ve yeni yüz tanımlacamk için aşağıdaki kodu kullanıyoruz. Her biri için oluşturduğumuz farklı açıklamalar ve farklı görüntüleri 
      //tüm bunlar geri döndürecek 
      return new faceapi.LabeledFaceDescriptors(label, descriptions)
    })
  )
}
