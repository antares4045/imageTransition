примерный ход мысли отсюда: https://github.com/foo52ru/img2img_tensorflow

сначала в __firstLearn.py__ в переменной `imageName` задаём картинку, из которой трансформация происходит и запускаем __firstLearn.py__

появляются папки models-first и repaired-first =>
в repaired выбираем понравившееся качество картинки и модель с соответствующим индексом устанавливаем в переменную `firstModel` из __secondLearn.py__ там же в `imageName` заносим картинку для финала трансформации и запускаем __secondLearn.py__

появляются папки models-second и repaired-second =>  
в repaired выбираем понравившееся качество картинки и модель с соответствующим индексом устанавливаем в переменную `secondModelPath` из __generateResult.py__ (как и все остальные переменные в мейне там (да, там всё руками -- мне влом))

на выходе получаем гифку и видяшку трансформации в корне трансформации

![например такая гифка](out.gif)

ток у меня почему-то получается научить только круглые картинки -- буду думать (наверное)