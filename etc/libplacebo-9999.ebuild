# Copyright 1999-2017 Gentoo Foundation
# Distributed under the terms of the GNU General Public License v2

EAPI=6

inherit meson git-r3

DESCRIPTION="Reusable library for GPU-accelerated image processing primitives"
HOMEPAGE="https://github.com/haasn/libplacebo"
EGIT_REPO_URI="https://github.com/haasn/libplacebo"

LICENSE="LGPLv2.1+"
SLOT="0"
KEYWORDS=""
IUSE="vulkan"

DEPEND="vulkan? ( media-libs/vulkan-loader )"
RDEPEND="${DEPEND}"

DOCS="README.md"

src_configure() {
	local emesonargs=(
		-Dvulkan=$(usex vulkan true false)
	)
	meson_src_configure
}
