/*
 * This file is part of libplacebo.
 *
 * libplacebo is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * libplacebo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with libplacebo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "common.h"
#include <libplacebo/utils/dolbyvision.h>

#ifdef PL_HAVE_LIBDOVI
#include <libplacebo/tone_mapping.h>
#include <libdovi/rpu_parser.h>
#endif

void pl_hdr_metadata_from_dovi_rpu(struct pl_hdr_metadata *out,
                                   const uint8_t *buf, size_t size)
{
#ifdef PL_HAVE_LIBDOVI
    if (buf && size) {
        DoviRpuOpaque *rpu =
            dovi_parse_unspec62_nalu(buf, size);
        const DoviRpuDataHeader *header = dovi_rpu_get_header(rpu);

        if (header && header->vdr_dm_metadata_present_flag) {
            // Profile 4 reshaping isn't done as it is a dual layer format.
            // However there are still unknowns on its EOTF, so it cannot be enabled.
            //
            // For profile 7, the brightness metadata can still be used as most
            // titles are going to have accurate metadata<->image brightness,
            // with the exception of some titles that require the enhancement layer
            // to be processed to restore the intended brightness, which would then
            // match the metadata values.
            if (header->guessed_profile == 4) {
                goto done;
            }

            const DoviVdrDmData *vdr_dm_data = dovi_rpu_get_vdr_dm_data(rpu);
            if (vdr_dm_data->dm_data.level1) {
                const DoviExtMetadataBlockLevel1 *l1 = vdr_dm_data->dm_data.level1;
                const float max_luma =
                    pl_hdr_rescale(PL_HDR_PQ, PL_HDR_NITS, l1->max_pq / 4095.0f);

                for (int i = 0; i < 3; i++)
                    out->scene_max[i] = max_luma;

                out->scene_avg =
                    pl_hdr_rescale(PL_HDR_PQ, PL_HDR_NITS, l1->avg_pq / 4095.0f);
            }

            dovi_rpu_free_vdr_dm_data(vdr_dm_data);
        }

    done:
        dovi_rpu_free_header(header);
        dovi_rpu_free(rpu);
    }
#endif
}
