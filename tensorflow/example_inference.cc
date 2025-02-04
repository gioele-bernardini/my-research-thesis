#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Includi il file generato (assicurati che il nome della variabile corrisponda a quello definito nel file)
// Di solito, il file contiene una dichiarazione del tipo: "extern const unsigned char g_converted_model_int8[];" 
// e la sua dimensione "extern const int g_converted_model_int8_len;"
#include "model_data.cc"

// Imposta la dimensione dell'area di memoria (tensor arena). Questo valore potrebbe dover essere regolato in base al modello.
constexpr int kTensorArenaSize = 10 * 1024;  // ad esempio, 10 KB
uint8_t tensor_arena[kTensorArenaSize];

int main() {
  // Imposta un error reporter per TFLM.
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Carica il modello dall'array definito in model_data.cc.
  const tflite::Model* model = tflite::GetModel(g_converted_model_int8);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Schema del modello (%d) non corrisponde a TFLITE_SCHEMA_VERSION (%d).",
                           model->version(), TFLITE_SCHEMA_VERSION);
    return -1;
  }

  // Crea un resolver che registra tutte le operazioni (o solo quelle necessarie).
  static tflite::AllOpsResolver resolver;

  // Crea l'interprete.
  static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

  // Alloca i tensori necessari.
  TfLiteStatus allocate_status = interpreter.AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("Allocazione dei tensori fallita.");
    return -1;
  }

  // Ottieni il tensore di input (assumendo che il modello abbia almeno un input).
  TfLiteTensor* input = interpreter.input(0);

  // ** Imposta i dati di input **
  // Per esempio, se stai eseguendo una classificazione e hai già i dati quantizzati in INT8:
  // Nota: Assicurati che i dati di input siano nel formato e nella scala corretti!
  // Esempio: per un'immagine o uno spettrogramma, copia i dati nel tensore.
  // Supponiamo di avere un array "my_input_data" di dimensione pari a input->bytes.
  // Qui mostro come copiarlo:
  /*
  for (int i = 0; i < input->bytes; i++) {
    input->data.int8[i] = my_input_data[i];
  }
  */
  // Se non hai dati reali, puoi simulare un input (ad es. impostando tutto a 0).
  for (int i = 0; i < input->bytes; i++) {
    input->data.int8[i] = 0;
  }

  // Esegui l'inferenza.
  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Errore durante l'invocazione del modello.");
    return -1;
  }

  // Ottieni il tensore di output (assumendo che il modello abbia almeno un output).
  TfLiteTensor* output = interpreter.output(0);

  // ** Leggi i risultati **
  // Ad esempio, se il modello è un classificatore, potresti avere un array di punteggi per ogni classe.
  // Qui, stampiamo i primi 10 valori (o il numero di classi, se inferiore).
  int output_size = output->bytes;  // oppure usa output->dims->data[1] se conosci la dimensione specifica
  error_reporter->Report("Risultati inferenza:");
  for (int i = 0; i < output_size; i++) {
    error_reporter->Report("Valore %d: %d", i, output->data.int8[i]);
  }

  return 0;
}
