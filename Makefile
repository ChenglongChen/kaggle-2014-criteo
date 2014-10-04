all: gbdt fm

gbdt:
	make -C solvers/gbdt
	ln -sf solvers/gbdt/gbdt

fm:
	make -C solvers/fm
	ln -sf solvers/fm/fm

clean:
	rm -f gbdt fm
	make -C solvers/gbdt clean
	make -C solvers/fm clean
