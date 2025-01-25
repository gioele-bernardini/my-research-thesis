#include <stdint.h>

// ====== INCLUDI TUTTI I FILE HEADER GENERATI ======

// BatchNorm1 Layer 1
#include "weights_16bit/bn1_biases_float16.h"
#include "weights_16bit/bn1_running_mean_float16.h"
#include "weights_16bit/bn1_running_variance_float16.h"
#include "weights_16bit/bn1_scale_float16.h"
#include "weights_16bit/bn1_shift_float16.h"
#include "weights_16bit/bn1_weights_float16.h"

// BatchNorm2 Layer 2
#include "weights_16bit/bn2_biases_float16.h"
#include "weights_16bit/bn2_running_mean_float16.h"
#include "weights_16bit/bn2_running_variance_float16.h"
#include "weights_16bit/bn2_scale_float16.h"
#include "weights_16bit/bn2_shift_float16.h"
#include "weights_16bit/bn2_weights_float16.h"

// Layer 1 Binarizzato
#include "weights_16bit/l1_biases_binarized.h"
#include "weights_16bit/l1_weights_binarized.h"

// Layer 2 Binarizzato
#include "weights_16bit/l2_biases_binarized.h"
#include "weights_16bit/l2_weights_binarized.h"

// Layer 4 Float16
#include "weights_16bit/l4_biases_float16.h"
#include "weights_16bit/l4_weights_float16.h"

// ====== FUNZIONE PER FORZARE IL LINK DEI PESI ======

/**
 * @brief Forza il linker a includere tutti gli array dei pesi e dei parametri BN.
 *        Questa funzione "tocca" ogni elemento degli array per evitare che il compilatore li elimini.
 */
void test_memory_usage(void)
{
    // Variabile volatile per impedire l'ottimizzazione completa
    volatile uint32_t dummy_sum = 0;

    // ----- BatchNorm1 Layer 1 -----
    for (unsigned int i = 0; i < bn1_biases_float16_len; i++) {
        dummy_sum += bn1_biases_float16[i];
    }
    for (unsigned int i = 0; i < bn1_running_mean_float16_len; i++) {
        dummy_sum += bn1_running_mean_float16[i];
    }
    for (unsigned int i = 0; i < bn1_running_variance_float16_len; i++) {
        dummy_sum += bn1_running_variance_float16[i];
    }
    for (unsigned int i = 0; i < bn1_scale_float16_len; i++) {
        dummy_sum += bn1_scale_float16[i];
    }
    for (unsigned int i = 0; i < bn1_shift_float16_len; i++) {
        dummy_sum += bn1_shift_float16[i];
    }
    for (unsigned int i = 0; i < bn1_weights_float16_len; i++) {
        dummy_sum += bn1_weights_float16[i];
    }

    // ----- BatchNorm2 Layer 2 -----
    for (unsigned int i = 0; i < bn2_biases_float16_len; i++) {
        dummy_sum += bn2_biases_float16[i];
    }
    for (unsigned int i = 0; i < bn2_running_mean_float16_len; i++) {
        dummy_sum += bn2_running_mean_float16[i];
    }
    for (unsigned int i = 0; i < bn2_running_variance_float16_len; i++) {
        dummy_sum += bn2_running_variance_float16[i];
    }
    for (unsigned int i = 0; i < bn2_scale_float16_len; i++) {
        dummy_sum += bn2_scale_float16[i];
    }
    for (unsigned int i = 0; i < bn2_shift_float16_len; i++) {
        dummy_sum += bn2_shift_float16[i];
    }
    for (unsigned int i = 0; i < bn2_weights_float16_len; i++) {
        dummy_sum += bn2_weights_float16[i];
    }

    // ----- Layer 1 Binarizzato -----
    for (unsigned int i = 0; i < l1_biases_binarized_len; i++) {
        dummy_sum += l1_biases_binarized[i];
    }
    for (unsigned int i = 0; i < l1_weights_binarized_len; i++) {
        dummy_sum += l1_weights_binarized[i];
    }

    // ----- Layer 2 Binarizzato -----
    for (unsigned int i = 0; i < l2_biases_binarized_len; i++) {
        dummy_sum += l2_biases_binarized[i];
    }
    for (unsigned int i = 0; i < l2_weights_binarized_len; i++) {
        dummy_sum += l2_weights_binarized[i];
    }

    // ----- Layer 4 Float16 -----
    for (unsigned int i = 0; i < l4_biases_float16_len; i++) {
        dummy_sum += l4_biases_float16[i];
    }
    for (unsigned int i = 0; i < l4_weights_float16_len; i++) {
        dummy_sum += l4_weights_float16[i];
    }

    // La variabile dummy_sum è volatile, quindi il compilatore non la rimuoverà
    // Nessuna azione necessaria con dummy_sum
}
